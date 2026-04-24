//! ONNX-backed embedder using [ort](https://ort.pyke.io).
//!
//! Loads a BERT-family embedding model (default: Jina v5 nano retrieval)
//! from a local directory or by downloading + caching via the HuggingFace
//! Hub API. Inference steps:
//!   1. Prepend the task prefix (`Document: ` / `Query: `) to each input.
//!   2. Tokenize the batch with the model's own `tokenizer.json`.
//!   3. Pad to the batch's max length, emit `input_ids` + `attention_mask`
//!      tensors, run the ONNX session.
//!   4. Mean-pool the `last_hidden_state` using the attention mask so
//!      padding tokens don't drag the centroid.
//!   5. L2-normalize so cosine search reduces to an inner product.

use super::Embedder;
use anyhow::{anyhow, Context, Result};
use ndarray::Array1;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use tokenizers::Tokenizer;

/// HuggingFace repo + specific ONNX file we ship against by default.
pub const DEFAULT_REPO: &str = "jinaai/jina-embeddings-v5-text-nano-retrieval";
pub const DEFAULT_ONNX_FILE: &str = "onnx/model_quantized.onnx";
/// Jina v5 retrieval adapter expects this prefix at indexing time.
pub const DEFAULT_DOC_PREFIX: &str = "Document: ";
/// Jina v5 nano retrieval is 768-dimensional.
pub const DEFAULT_DIM: usize = 768;

pub struct OrtEmbedder {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    dim: usize,
    prefix: String,
}

impl OrtEmbedder {
    /// Load from a local directory containing `model.onnx` (or the
    /// `onnx_filename` override) and `tokenizer.json`.
    pub fn from_dir(
        dir: impl AsRef<Path>,
        onnx_filename: &str,
        dim: usize,
        prefix: impl Into<String>,
    ) -> Result<Self> {
        let dir = dir.as_ref();
        let model_path = dir.join(onnx_filename);
        let tokenizer_path = dir.join("tokenizer.json");
        Self::from_paths(&model_path, &tokenizer_path, dim, prefix)
    }

    /// Load with explicit model + tokenizer paths.
    pub fn from_paths(
        model_path: &Path,
        tokenizer_path: &Path,
        dim: usize,
        prefix: impl Into<String>,
    ) -> Result<Self> {
        let session = Session::builder()
            .map_err(|e| anyhow!("creating ort session builder: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow!("setting ort optimization level: {e}"))?
            .commit_from_file(model_path)
            .map_err(|e| anyhow!("loading ONNX model from {}: {e}", model_path.display()))?;
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow!("loading tokenizer from {}: {e}", tokenizer_path.display()))?;
        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            dim,
            prefix: prefix.into(),
        })
    }

    /// Download (or reuse cached) model + tokenizer from HuggingFace Hub.
    pub fn from_hf_hub(
        repo: &str,
        onnx_file: &str,
        dim: usize,
        prefix: impl Into<String>,
    ) -> Result<Self> {
        let api = hf_hub::api::sync::Api::new().context("creating hf-hub API client")?;
        let r = api.model(repo.to_string());
        let model_path: PathBuf = r
            .get(onnx_file)
            .with_context(|| format!("downloading {repo}:{onnx_file}"))?;
        // Quantized ONNX files often ship alongside a `.onnx_data`
        // external-data sidecar; fetch it if present (ignore errors —
        // not every quantization variant has one).
        let _ = r.get(&format!("{onnx_file}_data"));
        let tokenizer_path: PathBuf = r
            .get("tokenizer.json")
            .with_context(|| format!("downloading {repo}:tokenizer.json"))?;
        Self::from_paths(&model_path, &tokenizer_path, dim, prefix)
    }

    /// Convenience: Jina v5 nano retrieval with default document prefix.
    pub fn jina_v5_nano_retrieval() -> Result<Self> {
        Self::from_hf_hub(
            DEFAULT_REPO,
            DEFAULT_ONNX_FILE,
            DEFAULT_DIM,
            DEFAULT_DOC_PREFIX,
        )
    }
}

impl Embedder for OrtEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Prefix every input. Done via owned Strings so we can feed
        // &[&str] to the tokenizer without fighting lifetimes.
        let prefixed: Vec<String> = texts
            .iter()
            .map(|t| format!("{}{t}", self.prefix))
            .collect();
        let prefixed_refs: Vec<&str> = prefixed.iter().map(String::as_str).collect();

        let encodings = self
            .tokenizer
            .encode_batch(prefixed_refs, true)
            .map_err(|e| anyhow!("tokenize batch: {e}"))?;

        let batch = encodings.len();
        let seq_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0);
        if seq_len == 0 {
            return Err(anyhow!("tokenizer produced empty sequences"));
        }

        // Pad manually to the longest sequence in the batch. The ORT
        // runtime expects a rectangular tensor. Flattened row-major.
        let total = batch * seq_len;
        let mut input_ids: Vec<i64> = vec![0; total];
        let mut attention_mask: Vec<i64> = vec![0; total];
        for (b, enc) in encodings.iter().enumerate() {
            let base = b * seq_len;
            for (t, &id) in enc.get_ids().iter().enumerate() {
                input_ids[base + t] = id as i64;
            }
            for (t, &m) in enc.get_attention_mask().iter().enumerate() {
                attention_mask[base + t] = m as i64;
            }
        }
        let attention_mask_for_pool = attention_mask.clone();

        let mut session = self.session.lock().map_err(|e| anyhow!("ort mutex: {e}"))?;
        let shape = vec![batch as i64, seq_len as i64];
        let ids_tensor = Tensor::from_array((shape.clone(), input_ids))
            .map_err(|e| anyhow!("building input_ids tensor: {e}"))?;
        let mask_tensor = Tensor::from_array((shape, attention_mask))
            .map_err(|e| anyhow!("building attention_mask tensor: {e}"))?;
        let outputs = session
            .run(ort::inputs!["input_ids" => ids_tensor, "attention_mask" => mask_tensor])
            .map_err(|e| anyhow!("ort session.run: {e}"))?;

        // BERT-family outputs `last_hidden_state`. If Jina changes
        // the output name in a future model, bump this constant and
        // document the transition.
        const OUTPUT_NAME: &str = "last_hidden_state";
        let tensor_ref = outputs
            .get(OUTPUT_NAME)
            .ok_or_else(|| anyhow!("model output missing {OUTPUT_NAME}"))?;
        let (shape, data) = tensor_ref
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("extracting {OUTPUT_NAME}: {e}"))?;

        if shape.len() != 3 {
            return Err(anyhow!(
                "expected last_hidden_state rank 3, got shape {shape:?}"
            ));
        }
        let (b, s, h) = (shape[0] as usize, shape[1] as usize, shape[2] as usize);
        if b != batch || s != seq_len {
            return Err(anyhow!(
                "model output shape ({b},{s},{h}) doesn't match input ({batch},{seq_len},-)"
            ));
        }
        if h != self.dim {
            return Err(anyhow!(
                "model returned hidden size {h}, expected {}",
                self.dim
            ));
        }

        // Mean-pool with attention mask.
        let mut out: Vec<Vec<f32>> = Vec::with_capacity(batch);
        for bi in 0..batch {
            let mut pooled = Array1::<f32>::zeros(h);
            let mut mass: f32 = 0.0;
            for si in 0..seq_len {
                let m = attention_mask_for_pool[bi * seq_len + si] as f32;
                if m == 0.0 {
                    continue;
                }
                mass += m;
                let base = bi * s * h + si * h;
                for (hi, p) in pooled.iter_mut().enumerate() {
                    *p += data[base + hi] * m;
                }
            }
            if mass > 0.0 {
                pooled /= mass;
            }
            // L2 normalize.
            let norm = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > f32::EPSILON {
                pooled /= norm;
            }
            out.push(pooled.to_vec());
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    // Real-model tests are gated with #[ignore] so CI doesn't download
    // ~80MB on every run. Local dev can run with:
    //   cargo test embed::ort -- --ignored
    // Unit tests that don't need a model run unconditionally.

    use super::*;

    #[test]
    fn default_constants_are_consistent() {
        // Guard against accidental mismatch between DIM and the repo
        // variant we ship against.
        assert_eq!(DEFAULT_DIM, 768);
        assert!(DEFAULT_REPO.contains("jina-embeddings-v5"));
        assert!(DEFAULT_ONNX_FILE.ends_with(".onnx"));
    }

    #[test]
    #[ignore = "downloads ~80MB ONNX model from HuggingFace; run with --ignored"]
    fn jina_v5_nano_retrieval_round_trip() {
        let e = OrtEmbedder::jina_v5_nano_retrieval().expect("load model");
        let out = e.embed(&["hello world", "goodbye world"]).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), DEFAULT_DIM);
        // Unit-normalized.
        for v in &out {
            let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((n - 1.0).abs() < 1e-4, "not unit: {n}");
        }
        // Deterministic for same input.
        let again = e.embed(&["hello world"]).unwrap();
        assert_eq!(out[0], again[0]);
    }

    #[test]
    #[ignore = "downloads ~80MB ONNX model from HuggingFace; run with --ignored"]
    fn semantically_similar_cosine_exceeds_unrelated() {
        let e = OrtEmbedder::jina_v5_nano_retrieval().expect("load model");
        let out = e
            .embed(&[
                "the cat sat on the mat",
                "a feline rested on the rug",
                "quarterly financial reporting deadlines",
            ])
            .unwrap();
        let cos = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        let related = cos(&out[0], &out[1]);
        let unrelated = cos(&out[0], &out[2]);
        assert!(
            related > unrelated + 0.05,
            "related={related} not clearly above unrelated={unrelated}"
        );
    }
}
