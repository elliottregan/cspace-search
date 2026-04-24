//! Indexer: embed records, upsert to a vector store, delete orphans.
//! Corpus-agnostic. Ports `search/index/index.go`.
//!
//! Storage: embedded sqlite with the [sqlite-vec] extension loaded.
//! Replaces the Go version's HTTP client to qdrant — no daemon, no
//! Docker, one sqlite file per project holds every corpus's vectors.
//!
//! [sqlite-vec]: https://github.com/asg017/sqlite-vec

// Phase 3 lands storage + Upserter; Phase 4 adds the Embedder and
// Phase 5 wires Config + run() into the init subcommand.
#![allow(dead_code)]

use std::collections::BTreeMap;

pub mod sqlite;

/// Minimal vector+payload+id the indexer hands to [`Upserter::upsert_points`].
#[derive(Debug, Clone)]
pub struct Point {
    pub id: u64,
    pub vector: Vec<f32>,
    pub payload: BTreeMap<String, serde_json::Value>,
}

/// Writes points to a vector store and reports what's already there,
/// so the indexer can skip unchanged records and delete orphans.
pub trait Upserter {
    /// Create the collection if missing. `dim` is the embedding
    /// dimensionality (e.g. 768 for Jina v5 nano).
    fn ensure_collection(&self, name: &str, dim: usize) -> anyhow::Result<()>;

    /// Upsert a batch of points. `progress` is called after each
    /// internal batch with `(done, total)`.
    fn upsert_points(
        &self,
        collection: &str,
        points: &[Point],
        batch_size: usize,
        progress: Option<&dyn Fn(usize, usize)>,
    ) -> anyhow::Result<()>;

    /// Map of `id → content_hash` for every row in the collection.
    /// Used by the indexer for change detection.
    fn existing_points(&self, collection: &str) -> anyhow::Result<BTreeMap<u64, String>>;

    /// Remove points by id. No-op when `ids` is empty.
    fn delete_points(&self, collection: &str, ids: &[u64]) -> anyhow::Result<()>;
}
