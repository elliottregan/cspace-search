//! Embedder: turn text into unit-normalized vectors for the indexer.
//!
//! Ports the Go `index.Embedder` interface. Phase 4a (this commit)
//! lands the trait + a deterministic `FakeEmbedder` so the indexer
//! loop (Phase 5) can be written and tested without a real model.
//! Phase 4b will add the real candle-backed implementation once the
//! model decision (v2-via-candle-native vs v5-via-llama-cpp-2 vs
//! port-v5-to-candle) lands.

#![allow(dead_code)]

use sha2::{Digest, Sha256};

/// Embedder embeds texts in batches and returns unit vectors.
/// Trait is sync for now; Phase 4b can switch to an `async_trait`
/// if the candle inference path benefits from it.
pub trait Embedder: Send + Sync {
    /// Output dimensionality. Used by the indexer to size the
    /// sqlite-vec virtual-table column.
    fn dim(&self) -> usize;

    /// Embed a batch of texts. Returned vectors are unit-normalized.
    /// Length of the returned vec equals the input slice length; the
    /// indexer checks this and treats a mismatch as a hard error.
    fn embed(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>>;
}

/// Deterministic content-addressed embedder for tests and for wiring
/// the indexer loop before Phase 4b. Produces unit vectors derived
/// from sha256 of the input, so repeated embeds of the same text
/// produce byte-identical vectors — exactly what change detection
/// needs to exercise.
///
/// Not for production use: vectors have no semantic meaning.
pub struct FakeEmbedder {
    dim: usize,
}

impl FakeEmbedder {
    pub fn new(dim: usize) -> Self {
        assert!(dim > 0, "FakeEmbedder dim must be > 0");
        Self { dim }
    }
}

impl Embedder for FakeEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    fn embed(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| fake_vec(t, self.dim)).collect())
    }
}

fn fake_vec(text: &str, dim: usize) -> Vec<f32> {
    // Seed a vector of floats from sha256(text), then normalize.
    let mut h = Sha256::new();
    h.update(text.as_bytes());
    let sum = h.finalize();
    let mut v: Vec<f32> = (0..dim)
        .map(|i| {
            // Pull 4 bytes from the hash (wrapping), interpret as u32,
            // scale to [-1, 1). Sufficient distinctness for tests.
            let b = i * 4;
            let word = u32::from_le_bytes([
                sum[b % sum.len()],
                sum[(b + 1) % sum.len()],
                sum[(b + 2) % sum.len()],
                sum[(b + 3) % sum.len()],
            ]);
            ((word as f64) / (u32::MAX as f64) * 2.0 - 1.0) as f32
        })
        .collect();
    let norm = v
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt()
        .max(f32::EPSILON);
    for x in &mut v {
        *x /= norm;
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_unit(v: &[f32]) -> bool {
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        (n - 1.0).abs() < 1e-5
    }

    #[test]
    fn fake_embedder_is_deterministic() {
        let e = FakeEmbedder::new(768);
        let a = e.embed(&["hello world"]).unwrap();
        let b = e.embed(&["hello world"]).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn fake_embedder_differs_by_input() {
        let e = FakeEmbedder::new(32);
        let out = e.embed(&["alpha", "beta"]).unwrap();
        assert_ne!(out[0], out[1]);
    }

    #[test]
    fn fake_embedder_returns_unit_vectors() {
        let e = FakeEmbedder::new(16);
        let out = e.embed(&["x", "yy", "zzz"]).unwrap();
        for v in &out {
            assert_eq!(v.len(), 16);
            assert!(is_unit(v), "vector not unit-normalized: {v:?}");
        }
    }

    #[test]
    fn fake_embedder_preserves_batch_length() {
        let e = FakeEmbedder::new(8);
        let out = e.embed(&["a", "b", "c"]).unwrap();
        assert_eq!(out.len(), 3);
    }
}
