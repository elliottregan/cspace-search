//! llama.cpp-backed embedder.
//!
//! Reads GGUF models natively — default target is the same
//! `v5-nano-retrieval-Q8_0.gguf` the Go version used, via
//! [`llama-cpp-2`](https://crates.io/crates/llama-cpp-2). Pros:
//! - No third-party CDN at build time (ORT's pyke.io outage blocked
//!   the pure-ONNX path).
//! - Vendored llama.cpp C++ builds everywhere Cargo runs.
//! - Jina v5's architecture (EuroBERT derivative) loads unchanged
//!   via the GGUF export the Jina team ships.
//!
//! Inference:
//!   1. Prepend the task prefix (`Document: ` for indexing).
//!   2. Tokenize via the model's own vocab.
//!   3. Add each sequence to a `LlamaBatch` with its own `seq_id`.
//!   4. `context.decode(&mut batch)`.
//!   5. Pull per-sequence pooled embeddings via
//!      `embeddings_seq_ith(seq_id)` — pooling type is set to MEAN
//!      on the context.
//!   6. L2-normalize so cosine search reduces to an inner product.

use super::Embedder;
use anyhow::{anyhow, Context, Result};
use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

/// HuggingFace repo + GGUF file we ship against by default.
pub const DEFAULT_REPO: &str = "jinaai/jina-embeddings-v5-text-nano-retrieval";
pub const DEFAULT_GGUF_FILE: &str = "v5-nano-retrieval-Q8_0.gguf";
/// Jina v5 retrieval adapter expects this prefix at indexing time.
pub const DEFAULT_DOC_PREFIX: &str = "Document: ";
/// Jina v5 retrieval adapter expects this prefix at query time.
pub const DEFAULT_QUERY_PREFIX: &str = "Query: ";
/// Jina v5 nano retrieval is 768-dimensional.
pub const DEFAULT_DIM: usize = 768;
/// Default context window. Jina v5 supports up to 8192 tokens; our
/// embed texts are truncated at ~12K chars ≈ 3K tokens, so 4096 is
/// plenty for a batch of documents.
pub const DEFAULT_N_CTX: u32 = 4096;

pub struct LlamaEmbedder {
    // Backend must outlive any context. We hold an Arc so clones of
    // this embedder (should we ever want them) share one instance.
    _backend: Arc<LlamaBackend>,
    model: LlamaModel,
    ctx_params: LlamaContextParams,
    dim: usize,
    /// Prefix prepended to document text in `embed`.
    doc_prefix: String,
    /// Prefix prepended to query text in `embed_query`.
    query_prefix: String,
    // Serialize calls through a mutex: llama contexts aren't `Sync`,
    // and the model/context pair isn't reusable across threads
    // without external locking. Embedding throughput is batch-bound,
    // not contention-bound, so the mutex is fine.
    lock: Mutex<()>,
}

/// Process-wide llama backend. `LlamaBackend::init()` is not safe to
/// call twice (errors on repeat init on some platforms), so every
/// embedder instance — including concurrent ones in tests — must
/// share the same handle.
fn shared_backend() -> Result<Arc<LlamaBackend>> {
    static BACKEND: OnceLock<std::result::Result<Arc<LlamaBackend>, String>> = OnceLock::new();
    match BACKEND.get_or_init(|| {
        LlamaBackend::init()
            .map(Arc::new)
            .map_err(|e| format!("initializing llama backend: {e}"))
    }) {
        Ok(b) => Ok(b.clone()),
        Err(msg) => Err(anyhow!("{msg}")),
    }
}

impl LlamaEmbedder {
    /// Load a GGUF model from a local path with explicit prefixes.
    pub fn from_path(
        path: &Path,
        dim: usize,
        doc_prefix: impl Into<String>,
        query_prefix: impl Into<String>,
    ) -> Result<Self> {
        let backend = shared_backend()?;
        let model_params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(&backend, path, &model_params)
            .with_context(|| format!("loading GGUF model from {}", path.display()))?;

        let ctx_params = LlamaContextParams::default()
            .with_embeddings(true)
            .with_pooling_type(LlamaPoolingType::Mean)
            .with_n_ctx(NonZeroU32::new(DEFAULT_N_CTX));

        Ok(Self {
            _backend: backend,
            model,
            ctx_params,
            dim,
            doc_prefix: doc_prefix.into(),
            query_prefix: query_prefix.into(),
            lock: Mutex::new(()),
        })
    }

    /// Download (or reuse cached) GGUF from HuggingFace Hub.
    pub fn from_hf_hub(
        repo: &str,
        gguf_file: &str,
        dim: usize,
        doc_prefix: impl Into<String>,
        query_prefix: impl Into<String>,
    ) -> Result<Self> {
        let api = hf_hub::api::sync::Api::new().context("creating hf-hub API client")?;
        let r = api.model(repo.to_string());
        let path: PathBuf = r
            .get(gguf_file)
            .with_context(|| format!("downloading {repo}:{gguf_file}"))?;
        Self::from_path(&path, dim, doc_prefix, query_prefix)
    }

    /// Convenience: Jina v5 nano retrieval with the Q8_0 GGUF and
    /// the standard `Document: ` / `Query: ` prefix pair.
    pub fn jina_v5_nano_retrieval() -> Result<Self> {
        Self::from_hf_hub(
            DEFAULT_REPO,
            DEFAULT_GGUF_FILE,
            DEFAULT_DIM,
            DEFAULT_DOC_PREFIX,
            DEFAULT_QUERY_PREFIX,
        )
    }

    /// Shared inference core. Takes already-prefixed texts and runs
    /// the full tokenize → batch → decode → pool → L2-normalize
    /// pipeline.
    fn embed_prefixed(&self, prefixed: &[String]) -> Result<Vec<Vec<f32>>> {
        if prefixed.is_empty() {
            return Ok(Vec::new());
        }
        let _guard = self.lock.lock().map_err(|e| anyhow!("llama mutex: {e}"))?;

        let mut tokenized: Vec<Vec<llama_cpp_2::token::LlamaToken>> =
            Vec::with_capacity(prefixed.len());
        for text in prefixed {
            let toks = self
                .model
                .str_to_token(text, AddBos::Always)
                .map_err(|e| anyhow!("tokenize: {e}"))?;
            if toks.is_empty() {
                return Err(anyhow!("tokenizer returned zero tokens"));
            }
            tokenized.push(toks);
        }

        let n_ctx: usize = self
            .ctx_params
            .n_ctx()
            .map(|n| usize::try_from(n.get()).unwrap_or(usize::MAX))
            .unwrap_or(DEFAULT_N_CTX as usize);
        for (i, toks) in tokenized.iter().enumerate() {
            if toks.len() > n_ctx {
                return Err(anyhow!(
                    "text {i} tokenized to {} tokens, exceeds context window {n_ctx}",
                    toks.len()
                ));
            }
        }

        let total_tokens: usize = tokenized.iter().map(Vec::len).sum();
        let n_seq_max = i32::try_from(tokenized.len())
            .map_err(|_| anyhow!("too many sequences in one batch"))?;
        let mut batch = LlamaBatch::new(total_tokens, n_seq_max);
        for (seq_id, toks) in tokenized.iter().enumerate() {
            batch
                .add_sequence(toks, seq_id as i32, false)
                .map_err(|e| anyhow!("batch.add_sequence(seq={seq_id}): {e}"))?;
        }

        // LlamaContextParams defaults n_seq_max to 1; size the slot
        // count to this batch.
        let ctx_params = self
            .ctx_params
            .clone()
            .with_n_seq_max(tokenized.len() as u32);
        let mut ctx = self
            .model
            .new_context(&self._backend, ctx_params)
            .map_err(|e| anyhow!("creating llama context: {e}"))?;
        ctx.decode(&mut batch)
            .map_err(|e| anyhow!("llama decode: {e}"))?;

        let mut out: Vec<Vec<f32>> = Vec::with_capacity(tokenized.len());
        for seq_id in 0..tokenized.len() {
            let emb = ctx
                .embeddings_seq_ith(seq_id as i32)
                .map_err(|e| anyhow!("embeddings_seq_ith(seq={seq_id}): {e}"))?;
            if emb.len() != self.dim {
                return Err(anyhow!(
                    "model returned {} dims, expected {}",
                    emb.len(),
                    self.dim
                ));
            }
            let mut v = emb.to_vec();
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > f32::EPSILON {
                for x in &mut v {
                    *x /= norm;
                }
            }
            out.push(v);
        }
        Ok(out)
    }
}

impl Embedder for LlamaEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let prefixed: Vec<String> = texts
            .iter()
            .map(|t| format!("{}{t}", self.doc_prefix))
            .collect();
        self.embed_prefixed(&prefixed)
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let prefixed = vec![format!("{}{text}", self.query_prefix)];
        let mut out = self.embed_prefixed(&prefixed)?;
        out.pop()
            .ok_or_else(|| anyhow!("embed_query returned no vector"))
    }
}

#[cfg(test)]
mod tests {
    // Real-model tests are gated with #[ignore] so CI doesn't download
    // the 80MB GGUF on every run. Exercise locally with:
    //   cargo test embed::llama -- --ignored

    use super::*;

    #[test]
    fn default_constants_are_consistent() {
        assert_eq!(DEFAULT_DIM, 768);
        assert!(DEFAULT_REPO.contains("jina-embeddings-v5"));
        assert!(DEFAULT_GGUF_FILE.ends_with(".gguf"));
    }

    #[test]
    #[ignore = "downloads ~80MB GGUF from HuggingFace; run with --ignored"]
    fn jina_v5_nano_retrieval_round_trip() {
        let e = LlamaEmbedder::jina_v5_nano_retrieval().expect("load model");
        let out = e.embed(&["hello world", "goodbye world"]).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), DEFAULT_DIM);
        for v in &out {
            let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((n - 1.0).abs() < 1e-4, "not unit: {n}");
        }
        let again = e.embed(&["hello world"]).unwrap();
        // Inference is numerically stable but not bit-identical across
        // different batch sizes: the GPU reduction order changes when
        // the batch shape changes, producing ~1e-4 rounding deltas.
        // Semantically the vectors are the same; cosine should round
        // to 1.0 well within tolerance.
        let cos: f32 = out[0].iter().zip(&again[0]).map(|(a, b)| a * b).sum();
        assert!(
            (cos - 1.0).abs() < 1e-3,
            "repeat of same input should cosine ≈ 1.0, got {cos}"
        );
    }

    #[test]
    #[ignore = "downloads ~80MB GGUF from HuggingFace; run with --ignored"]
    fn semantically_similar_cosine_exceeds_unrelated() {
        let e = LlamaEmbedder::jina_v5_nano_retrieval().expect("load model");
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
