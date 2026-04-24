//! Query path: embed the query, kNN against the vector store, dedupe
//! by path, build the response envelope. Ports `search/query/query.go`.
//!
//! The store implementation (sqlite-vec) handles the kNN itself; this
//! module is corpus-agnostic plumbing that owns the top-k shaping.

#![allow(dead_code)]

use crate::corpus::Corpus;
use crate::embed::Embedder;
use crate::index::Searcher;
use anyhow::{Context, Result};
use serde::Serialize;
use std::path::Path;

/// Default result count when the caller passes zero.
pub const DEFAULT_TOP_K: usize = 10;
/// Upper bound on `top_k`. Prevents a pathological request from
/// pulling the entire store into memory.
pub const MAX_TOP_K: usize = 50;
/// The store is asked for `TOP_K * OVERFETCH` rows so dedupe-by-path
/// can evict duplicate chunks without running the search short.
pub const OVERFETCH: usize = 3;

#[derive(Debug, Clone, Serialize)]
pub struct Hit {
    pub path: String,
    pub line_start: u32,
    pub line_end: u32,
    pub score: f32,
    pub kind: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub content_hash: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub preview: String,
    #[serde(skip_serializing_if = "is_default_cluster")]
    pub cluster_id: i64,
}

fn is_default_cluster(v: &i64) -> bool {
    *v == 0
}

#[derive(Debug, Clone, Serialize)]
pub struct Envelope {
    pub query: String,
    pub corpus: String,
    pub results: Vec<Hit>,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub warning: String,
}

pub struct RunConfig<'a> {
    pub corpus: &'a dyn Corpus,
    pub embedder: &'a dyn Embedder,
    pub searcher: &'a dyn Searcher,
    pub project_root: &'a Path,
    pub query: &'a str,
    /// 0 → `DEFAULT_TOP_K`; >`MAX_TOP_K` is clamped.
    pub top_k: usize,
    /// Retain the indexer-assigned cluster id on hits. Default false —
    /// noise for most CLI users; future `cspace-search clusters`
    /// opts in.
    pub with_cluster: bool,
}

/// Execute a single query end-to-end.
pub fn run(cfg: RunConfig<'_>) -> Result<Envelope> {
    let top_k = clamp_top_k(cfg.top_k);
    let collection = cfg.corpus.collection(cfg.project_root);

    let vec = cfg.embedder.embed_query(cfg.query).context("embed query")?;
    if all_zero(&vec) {
        return Ok(Envelope {
            query: cfg.query.to_string(),
            corpus: cfg.corpus.id().to_string(),
            results: Vec::new(),
            warning: "query produced a degenerate embedding; try a more specific phrase".into(),
        });
    }

    let raws = cfg
        .searcher
        .search(&collection, &vec, top_k * OVERFETCH)
        .with_context(|| format!("search {collection}"))?;

    let mut hits: Vec<Hit> = raws.into_iter().map(hit_from_raw).collect();
    if !cfg.with_cluster {
        for h in &mut hits {
            h.cluster_id = 0;
        }
    }
    let mut deduped = dedupe_by_path(hits);
    deduped.truncate(top_k);

    Ok(Envelope {
        query: cfg.query.to_string(),
        corpus: cfg.corpus.id().to_string(),
        results: deduped,
        warning: String::new(),
    })
}

/// Collapse multiple chunks of the same path, keeping the
/// highest-scoring. Result is sorted by score descending.
pub fn dedupe_by_path(hits: Vec<Hit>) -> Vec<Hit> {
    use std::collections::HashMap;
    let mut best: HashMap<String, Hit> = HashMap::new();
    for h in hits {
        match best.get(&h.path) {
            Some(cur) if cur.score >= h.score => {}
            _ => {
                best.insert(h.path.clone(), h);
            }
        }
    }
    let mut out: Vec<Hit> = best.into_values().collect();
    out.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out
}

fn hit_from_raw(r: crate::index::RawHit) -> Hit {
    let mut h = Hit {
        path: String::new(),
        line_start: 0,
        line_end: 0,
        score: r.score,
        kind: String::new(),
        content_hash: String::new(),
        preview: String::new(),
        cluster_id: 0,
    };
    if let Some(v) = r.payload.get("path").and_then(|v| v.as_str()) {
        h.path = v.to_string();
    }
    if let Some(v) = r.payload.get("kind").and_then(|v| v.as_str()) {
        h.kind = v.to_string();
    }
    if let Some(v) = r.payload.get("line_start").and_then(|v| v.as_u64()) {
        h.line_start = v as u32;
    }
    if let Some(v) = r.payload.get("line_end").and_then(|v| v.as_u64()) {
        h.line_end = v as u32;
    }
    if let Some(v) = r.payload.get("content_hash").and_then(|v| v.as_str()) {
        h.content_hash = v.to_string();
    }
    if let Some(v) = r.payload.get("cluster_id").and_then(|v| v.as_i64()) {
        h.cluster_id = v;
    }
    h
}

fn clamp_top_k(k: usize) -> usize {
    if k == 0 {
        DEFAULT_TOP_K
    } else if k > MAX_TOP_K {
        MAX_TOP_K
    } else {
        k
    }
}

fn all_zero(v: &[f32]) -> bool {
    v.iter().all(|&x| x == 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hit(path: &str, score: f32) -> Hit {
        Hit {
            path: path.into(),
            line_start: 0,
            line_end: 0,
            score,
            kind: String::new(),
            content_hash: String::new(),
            preview: String::new(),
            cluster_id: 0,
        }
    }

    #[test]
    fn dedupe_keeps_best_score_per_path() {
        let hits = vec![hit("a.go", 0.5), hit("a.go", 0.8), hit("b.go", 0.6)];
        let got = dedupe_by_path(hits);
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].path, "a.go");
        assert!((got[0].score - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn dedupe_sorted_descending_by_score() {
        let hits = vec![hit("a.go", 0.3), hit("b.go", 0.9), hit("c.go", 0.6)];
        let got = dedupe_by_path(hits);
        assert_eq!(got.len(), 3);
        for w in got.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
    }

    #[test]
    fn clamp_top_k_defaults_and_caps() {
        assert_eq!(clamp_top_k(0), DEFAULT_TOP_K);
        assert_eq!(clamp_top_k(5), 5);
        assert_eq!(clamp_top_k(999), MAX_TOP_K);
    }

    /// End-to-end smoke: index a few context docs with the fake
    /// embedder, then query one of them back and expect the matching
    /// path to rank first. Uses FakeEmbedder so the query and doc
    /// vectors are byte-identical when the text matches.
    #[test]
    fn run_returns_top_match_for_known_text() {
        use crate::corpus::ContextCorpus;
        use crate::embed::FakeEmbedder;
        use crate::index::sqlite::SqliteUpserter;
        use crate::index::{self as idx, RunConfig as IdxRun};

        let dir = tempfile::tempdir().unwrap();
        let ctx = dir.path().join(".cspace/context");
        std::fs::create_dir_all(&ctx).unwrap();
        std::fs::write(ctx.join("principles.md"), "keep it simple").unwrap();
        std::fs::write(ctx.join("roadmap.md"), "ship in phases").unwrap();

        let corpus = ContextCorpus;
        let embedder = FakeEmbedder::new(32);
        let up = SqliteUpserter::in_memory().unwrap();

        idx::run(IdxRun {
            corpus: &corpus,
            embedder: &embedder,
            upserter: &up,
            project_root: dir.path(),
            batch_size: 8,
            progress: None,
        })
        .unwrap();

        // The fake embedder is content-addressed: identical text
        // produces identical vectors. Query with the exact embed_text
        // of principles.md to guarantee a perfect cosine hit.
        let embed_text = "Context (principles): .cspace/context/principles.md\n\nkeep it simple";
        let env = run(RunConfig {
            corpus: &corpus,
            embedder: &embedder,
            searcher: &up,
            project_root: dir.path(),
            query: embed_text,
            top_k: 3,
            with_cluster: false,
        })
        .unwrap();

        assert_eq!(env.corpus, "context");
        assert!(!env.results.is_empty());
        assert_eq!(env.results[0].path, ".cspace/context/principles.md");
    }
}
