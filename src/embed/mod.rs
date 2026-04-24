//! Embedder: turn text into unit-normalized vectors for the indexer.
//!
//! Ports the Go `index.Embedder` interface. Provides:
//! - [`FakeEmbedder`] — deterministic sha256-seeded vectors for tests.
//! - [`llama::LlamaEmbedder`] — GGUF-backed inference via llama.cpp,
//!   default target is Jina v5 nano retrieval.

#![allow(dead_code)]

pub mod llama;

use sha2::{Digest, Sha256};

/// Embedder embeds texts and returns unit vectors.
pub trait Embedder: Send + Sync {
    /// Output dimensionality. Used by the indexer to size the
    /// sqlite-vec virtual-table column.
    fn dim(&self) -> usize;

    /// Opaque identity of this embedder: model + config choices that
    /// affect the vector output. Stored per-collection by the store;
    /// on subsequent `init` runs, a mismatch triggers drop-and-rebuild
    /// so a model swap never silently leaves stale vectors in place.
    ///
    /// The string is compared for exact equality. It MUST:
    ///   - differ whenever a different embedder would produce
    ///     different vectors for the same input;
    ///   - be stable across process restarts with the same config;
    ///   - not depend on data outside the embedder's construction
    ///     (no timestamps, no host paths that might vary by machine).
    fn fingerprint(&self) -> String;

    /// Embed a batch of document texts. Returned vectors are
    /// unit-normalized. Length of the returned vec equals the input
    /// slice length; the indexer checks this and treats a mismatch as
    /// a hard error. Implementations may prepend a task prefix
    /// (Jina v5 uses `"Document: "` for indexing).
    fn embed(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>>;

    /// Embed a single query string. Jina-family retrieval models
    /// expect a different prefix (`"Query: "`) at query time than
    /// indexing time, so implementations route this through a
    /// separate codepath even though the underlying inference is the
    /// same shape as a batch-of-one.
    fn embed_query(&self, text: &str) -> anyhow::Result<Vec<f32>>;
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

    fn fingerprint(&self) -> String {
        // Dim fully determines the vector space; sha256 seeding is
        // fixed. No other knobs, so the dim alone is sufficient
        // identity.
        format!("fake:dim={}", self.dim)
    }

    fn embed(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| fake_vec(t, self.dim)).collect())
    }

    fn embed_query(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        // No prefix distinction for the fake — keep a query's vector
        // byte-identical to the doc-side vector of the same text, so
        // tests see cos == 1.0 for a round-trip.
        Ok(fake_vec(text, self.dim))
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
