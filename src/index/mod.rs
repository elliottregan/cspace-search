//! Indexer: embed records, upsert to a vector store, delete orphans.
//! Corpus-agnostic. Ports `search/index/index.go`.
//!
//! Storage: embedded sqlite with the [sqlite-vec] extension loaded.
//! Replaces the Go version's HTTP client to qdrant — no daemon, no
//! Docker, one sqlite file per project holds every corpus's vectors.
//!
//! [sqlite-vec]: https://github.com/asg017/sqlite-vec

#![allow(dead_code)]

use crate::corpus::{Corpus, Record};
use crate::embed::Embedder;
use anyhow::{anyhow, Context};
use std::collections::{BTreeMap, HashSet};
use std::path::Path;

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

/// One hit back from a kNN search: id + similarity score + the JSON
/// payload we stored alongside the vector.
#[derive(Debug, Clone)]
pub struct RawHit {
    pub id: u64,
    /// Similarity score in `[0, 1]`. Higher = closer. sqlite-vec
    /// returns L2 distance, so we convert via
    /// `score = 1 / (1 + distance)` for a monotonic, human-readable
    /// value; cosine ordering is preserved since all vectors are
    /// L2-normalized before insertion.
    pub score: f32,
    pub payload: std::collections::BTreeMap<String, serde_json::Value>,
}

/// Runs kNN queries against a vector store. Separate from
/// [`Upserter`] so the query codepath doesn't need the writer's
/// mutation surface (and vice versa). A single concrete type can
/// implement both — `SqliteUpserter` does.
pub trait Searcher {
    /// Nearest-neighbor search. `top_k` is a soft limit: callers
    /// typically over-request (e.g. `top_k * 3`) and dedupe
    /// afterwards, so implementations SHOULD return exactly `top_k`
    /// rows if that many exist.
    fn search(&self, collection: &str, vector: &[f32], top_k: usize)
        -> anyhow::Result<Vec<RawHit>>;
}

/// Default embedding batch size. Matches the Go version.
pub const DEFAULT_BATCH_SIZE: usize = 32;

/// Configuration for a single `run` of the indexer.
pub struct RunConfig<'a> {
    pub corpus: &'a dyn Corpus,
    pub embedder: &'a dyn Embedder,
    pub upserter: &'a dyn Upserter,
    pub project_root: &'a Path,
    /// How many records to embed per forward pass. Falls back to
    /// [`DEFAULT_BATCH_SIZE`] when zero.
    pub batch_size: usize,
    /// Optional callback fired once per upserted batch with
    /// `(done, total)`. Counts points embedded + written in this run;
    /// unchanged records skipped via the content-hash cache are not
    /// reflected in `total`.
    pub progress: Option<&'a dyn Fn(usize, usize)>,
}

/// Result summary for one indexer run.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct RunStats {
    /// Records the corpus produced.
    pub enumerated: usize,
    /// Records that survived the content-hash skip cache and went
    /// through the embedder.
    pub embedded: usize,
    /// Records dropped because they no longer appear in the corpus.
    pub orphans_deleted: usize,
}

/// One end-to-end index pass: enumerate → skip unchanged → embed
/// changed records → upsert → delete orphans. Ports `search/index/index.go:Run`.
pub fn run(cfg: RunConfig<'_>) -> anyhow::Result<RunStats> {
    let batch_size = if cfg.batch_size == 0 {
        DEFAULT_BATCH_SIZE
    } else {
        cfg.batch_size
    };
    let dim = cfg.embedder.dim();
    let collection = cfg.corpus.collection(cfg.project_root);

    cfg.upserter
        .ensure_collection(&collection, dim)
        .with_context(|| format!("ensure collection {collection}"))?;
    let existing = cfg
        .upserter
        .existing_points(&collection)
        .with_context(|| format!("list existing points in {collection}"))?;

    let records: Vec<Record> = cfg
        .corpus
        .enumerate(cfg.project_root)
        .with_context(|| format!("enumerate corpus {}", cfg.corpus.id()))?;
    let enumerated = records.len();

    // Partition into (seen_ids, records_needing_embedding). We skip any
    // record whose content hash matches what's already stored.
    let mut seen: HashSet<u64> = HashSet::with_capacity(enumerated);
    let mut to_embed: Vec<Record> = Vec::new();
    for rec in records {
        let id = rec.id();
        seen.insert(id);
        match existing.get(&id) {
            Some(stored_hash)
                if !rec.content_hash.is_empty() && stored_hash == &rec.content_hash =>
            {
                // Unchanged; skip.
            }
            _ => to_embed.push(rec),
        }
    }
    let embedded_total = to_embed.len();

    // Embed + upsert in batches. The embedder batches are the same size
    // as the upsert batches so the progress callback ticks at a uniform
    // granularity.
    let mut done = 0usize;
    for chunk in to_embed.chunks(batch_size) {
        let texts: Vec<&str> = chunk.iter().map(|r| r.embed_text.as_str()).collect();
        let vectors = cfg
            .embedder
            .embed(&texts)
            .with_context(|| format!("embed batch (offset {done})"))?;
        if vectors.len() != chunk.len() {
            return Err(anyhow!(
                "embedder returned {} vectors for {} texts",
                vectors.len(),
                chunk.len()
            ));
        }
        let points: Vec<Point> = chunk
            .iter()
            .zip(vectors)
            .map(|(rec, vec)| Point {
                id: rec.id(),
                vector: vec,
                payload: payload_for(rec),
            })
            .collect();
        cfg.upserter
            .upsert_points(&collection, &points, batch_size, None)
            .with_context(|| format!("upsert batch (offset {done})"))?;
        done += chunk.len();
        if let Some(cb) = cfg.progress {
            cb(done, embedded_total);
        }
    }

    // Orphan sweep — anything in the store that's not in the current
    // corpus enumeration gets evicted.
    let orphans: Vec<u64> = existing
        .keys()
        .copied()
        .filter(|id| !seen.contains(id))
        .collect();
    let orphans_deleted = orphans.len();
    if !orphans.is_empty() {
        cfg.upserter
            .delete_points(&collection, &orphans)
            .with_context(|| format!("delete orphans in {collection}"))?;
    }

    Ok(RunStats {
        enumerated,
        embedded: embedded_total,
        orphans_deleted,
    })
}

/// Build the payload stored alongside each embedding. Carries enough
/// metadata that a search hit is useful on its own (path + kind +
/// line range + hash) plus any corpus-specific `Record::extra` fields.
/// The `cluster_id = -1` default is reserved for a later clustering
/// pass that tags rows in place.
fn payload_for(r: &Record) -> BTreeMap<String, serde_json::Value> {
    let mut p: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    p.insert("path".into(), r.path.clone().into());
    p.insert("kind".into(), r.kind.clone().into());
    p.insert("line_start".into(), r.line_start.into());
    p.insert("line_end".into(), r.line_end.into());
    p.insert("content_hash".into(), r.content_hash.clone().into());
    p.insert("cluster_id".into(), (-1i64).into());
    for (k, v) in &r.extra {
        p.insert(k.clone(), v.clone());
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{CorpusConfig, PathGroupSpec};
    use crate::corpus::FileCorpus;
    use crate::embed::FakeEmbedder;
    use crate::index::sqlite::SqliteUpserter;
    use std::fs;

    fn seed_context(dir: &Path, files: &[(&str, &str)]) {
        let ctx = dir.join(".cspace").join("context");
        fs::create_dir_all(&ctx).unwrap();
        for (rel, body) in files {
            let p = ctx.join(rel);
            fs::create_dir_all(p.parent().unwrap()).unwrap();
            fs::write(p, body).unwrap();
        }
    }

    /// Test-local context corpus matching the default.yaml shape but
    /// opted on. Keeps the indexer tests independent of what the
    /// default ships.
    fn context_cfg() -> CorpusConfig {
        CorpusConfig {
            enabled: true,
            source: Some("filesystem".into()),
            type_name: Some("files".into()),
            record_kind: Some("context".into()),
            embed_header: Some("Context ({subkind}): {path}\n\n".into()),
            path_groups: vec![
                PathGroupSpec {
                    include: vec![
                        ".cspace/context/direction.md".into(),
                        ".cspace/context/principles.md".into(),
                        ".cspace/context/roadmap.md".into(),
                    ],
                    kind: Some("context".into()),
                    chunk: None,
                    extra: [("subkind".to_string(), "{basename_no_ext}".to_string())]
                        .into_iter()
                        .collect(),
                },
                PathGroupSpec {
                    include: vec![".cspace/context/findings/**/*.md".into()],
                    kind: Some("finding".into()),
                    chunk: None,
                    extra: std::collections::BTreeMap::new(),
                },
            ],
            ..CorpusConfig::default()
        }
    }

    #[test]
    fn first_run_embeds_and_upserts_all_records() {
        let dir = tempfile::tempdir().unwrap();
        seed_context(
            dir.path(),
            &[
                ("principles.md", "# P1"),
                ("roadmap.md", "# R1"),
                ("findings/a.md", "finding alpha"),
            ],
        );

        let corpus = FileCorpus::from_config("context", &context_cfg()).unwrap();
        let embedder = FakeEmbedder::new(16);
        let upserter = SqliteUpserter::in_memory().unwrap();

        let stats = run(RunConfig {
            corpus: &corpus,
            embedder: &embedder,
            upserter: &upserter,
            project_root: dir.path(),
            batch_size: 2,
            progress: None,
        })
        .unwrap();

        assert_eq!(stats.enumerated, 3);
        assert_eq!(stats.embedded, 3);
        assert_eq!(stats.orphans_deleted, 0);

        let collection = corpus.collection(dir.path());
        let existing = upserter.existing_points(&collection).unwrap();
        assert_eq!(existing.len(), 3);
    }

    #[test]
    fn second_run_skips_unchanged_and_embeds_only_changes() {
        let dir = tempfile::tempdir().unwrap();
        seed_context(
            dir.path(),
            &[
                ("principles.md", "# original"),
                ("roadmap.md", "# original"),
            ],
        );

        let corpus = FileCorpus::from_config("context", &context_cfg()).unwrap();
        let embedder = FakeEmbedder::new(16);
        let upserter = SqliteUpserter::in_memory().unwrap();

        let first = run(RunConfig {
            corpus: &corpus,
            embedder: &embedder,
            upserter: &upserter,
            project_root: dir.path(),
            batch_size: 8,
            progress: None,
        })
        .unwrap();
        assert_eq!(first.embedded, 2);

        // Edit one file; the other is unchanged.
        fs::write(dir.path().join(".cspace/context/principles.md"), "# edited").unwrap();

        let second = run(RunConfig {
            corpus: &corpus,
            embedder: &embedder,
            upserter: &upserter,
            project_root: dir.path(),
            batch_size: 8,
            progress: None,
        })
        .unwrap();
        assert_eq!(second.enumerated, 2);
        assert_eq!(second.embedded, 1, "only the edited file should re-embed");
        assert_eq!(second.orphans_deleted, 0);
    }

    #[test]
    fn removed_records_are_evicted_as_orphans() {
        let dir = tempfile::tempdir().unwrap();
        seed_context(
            dir.path(),
            &[
                ("principles.md", "# P"),
                ("findings/a.md", "alpha"),
                ("findings/b.md", "beta"),
            ],
        );

        let corpus = FileCorpus::from_config("context", &context_cfg()).unwrap();
        let embedder = FakeEmbedder::new(16);
        let upserter = SqliteUpserter::in_memory().unwrap();

        run(RunConfig {
            corpus: &corpus,
            embedder: &embedder,
            upserter: &upserter,
            project_root: dir.path(),
            batch_size: 8,
            progress: None,
        })
        .unwrap();

        // Remove one finding.
        fs::remove_file(dir.path().join(".cspace/context/findings/b.md")).unwrap();

        let stats = run(RunConfig {
            corpus: &corpus,
            embedder: &embedder,
            upserter: &upserter,
            project_root: dir.path(),
            batch_size: 8,
            progress: None,
        })
        .unwrap();
        assert_eq!(stats.orphans_deleted, 1);

        let existing = upserter
            .existing_points(&corpus.collection(dir.path()))
            .unwrap();
        assert_eq!(existing.len(), 2);
    }

    #[test]
    fn progress_callback_fires_per_upsert_batch() {
        let dir = tempfile::tempdir().unwrap();
        seed_context(
            dir.path(),
            &[
                ("principles.md", "a"),
                ("roadmap.md", "b"),
                ("direction.md", "c"),
                ("findings/1.md", "d"),
                ("findings/2.md", "e"),
            ],
        );

        let corpus = FileCorpus::from_config("context", &context_cfg()).unwrap();
        let embedder = FakeEmbedder::new(8);
        let upserter = SqliteUpserter::in_memory().unwrap();

        let calls = std::cell::RefCell::new(Vec::<(usize, usize)>::new());
        let cb = |d, t| calls.borrow_mut().push((d, t));
        run(RunConfig {
            corpus: &corpus,
            embedder: &embedder,
            upserter: &upserter,
            project_root: dir.path(),
            batch_size: 2,
            progress: Some(&cb),
        })
        .unwrap();

        // Five records, batch 2 → (2,5) (4,5) (5,5).
        let observed = calls.into_inner();
        assert_eq!(observed, vec![(2, 5), (4, 5), (5, 5)]);
    }

    #[test]
    fn empty_corpus_is_a_noop() {
        let dir = tempfile::tempdir().unwrap();
        // .cspace/context/ doesn't exist at all.
        let corpus = FileCorpus::from_config("context", &context_cfg()).unwrap();
        let embedder = FakeEmbedder::new(8);
        let upserter = SqliteUpserter::in_memory().unwrap();
        let stats = run(RunConfig {
            corpus: &corpus,
            embedder: &embedder,
            upserter: &upserter,
            project_root: dir.path(),
            batch_size: 4,
            progress: None,
        })
        .unwrap();
        assert_eq!(stats, RunStats::default());
    }
}
