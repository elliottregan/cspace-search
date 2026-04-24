//! Global embedding cache.
//!
//! A sqlite file at `~/.cspace-search/embed-cache.db` memoizes
//! `embed_text -> vector` outputs keyed by the inner embedder's
//! fingerprint. Hits return the cached vector without round-tripping
//! through the model.
//!
//! Where this earns its keep:
//!
//! - **Model-swap rebuilds.** When `ensure_collection` drops a stale
//!   vec0 table on fingerprint mismatch, the per-collection
//!   content-hash skip is wiped; every record re-embeds. With the
//!   cache warm (same fingerprint, same texts), the second+ rebuilds
//!   are near-free.
//! - **Multiple projects on the same machine.** Boilerplate text
//!   (license headers, repeated `Context (principles): …` snippets,
//!   identical commit messages after a squash) occurs across repos.
//! - **Interrupted indexing.** A `cspace-search init` that crashes
//!   partway leaves no per-collection state for records not yet
//!   upserted; the global cache captures what was embedded before the
//!   crash.
//!
//! Not a replacement for the per-collection content-hash skip — that
//! skip avoids *any* work for unchanged records, while this cache
//! still hashes + does a sqlite round-trip per record. Both live
//! side-by-side: hash-skip handles the common case, cache handles the
//! rebuild case.

use super::Embedder;
use anyhow::{anyhow, Context};
use rusqlite::Connection;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Sqlite-backed store for (fingerprint, text_hash) → vector. One
/// instance per process is expected; the lock+connection are private.
pub struct EmbedCache {
    conn: Mutex<Connection>,
    // Kept for diagnostics + tests that want to inspect the file.
    #[allow(dead_code)]
    path: PathBuf,
}

impl EmbedCache {
    /// Open (or create) the cache at `path`. Parent directories are
    /// created as needed. WAL mode lets concurrent `cspace-search`
    /// processes read + write without blocking each other for long.
    pub fn open(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).context("creating embed-cache parent dir")?;
        }
        let conn = Connection::open(&path).context("opening embed-cache db")?;
        conn.pragma_update(None, "journal_mode", "WAL")
            .context("enabling WAL on embed-cache")?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS embed_cache (
                fingerprint TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                vector BLOB NOT NULL,
                PRIMARY KEY (fingerprint, text_hash)
            )",
            [],
        )
        .context("ensuring embed_cache table")?;
        Ok(Self {
            conn: Mutex::new(conn),
            path,
        })
    }

    /// In-memory instance for tests.
    pub fn in_memory() -> anyhow::Result<Self> {
        let conn = Connection::open_in_memory().context("opening in-memory cache")?;
        conn.execute(
            "CREATE TABLE embed_cache (
                fingerprint TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                vector BLOB NOT NULL,
                PRIMARY KEY (fingerprint, text_hash)
            )",
            [],
        )
        .context("creating in-memory embed_cache")?;
        Ok(Self {
            conn: Mutex::new(conn),
            path: PathBuf::new(),
        })
    }

    /// Look up a cached vector. Returns `None` on miss, reserves
    /// errors for real IO/SQL failures so callers can distinguish.
    pub fn get(&self, fingerprint: &str, text: &str) -> anyhow::Result<Option<Vec<f32>>> {
        let hash = text_hash(text);
        let conn = self.conn.lock().map_err(|e| anyhow!("cache mutex: {e}"))?;
        let row: Option<Vec<u8>> = conn
            .query_row(
                "SELECT vector FROM embed_cache WHERE fingerprint = ?1 AND text_hash = ?2",
                rusqlite::params![fingerprint, hash],
                |r| r.get(0),
            )
            .map(Some)
            .or_else(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => Ok(None),
                other => Err(other),
            })
            .context("read embed_cache")?;
        Ok(row.map(|blob| bytes_to_f32(&blob)))
    }

    /// Insert (or overwrite) a (fingerprint, text) → vector mapping.
    /// Batch variant is preferred for throughput — use `put_many`.
    pub fn put(&self, fingerprint: &str, text: &str, vector: &[f32]) -> anyhow::Result<()> {
        self.put_many(fingerprint, &[(text, vector)])
    }

    /// Batch insert. All rows go through one transaction, so the
    /// caller pays one fsync for the whole batch rather than N.
    pub fn put_many(
        &self,
        fingerprint: &str,
        entries: &[(&str, &[f32])],
    ) -> anyhow::Result<()> {
        if entries.is_empty() {
            return Ok(());
        }
        let mut conn = self.conn.lock().map_err(|e| anyhow!("cache mutex: {e}"))?;
        let tx = conn.transaction().context("begin cache tx")?;
        {
            let mut stmt = tx.prepare_cached(
                "INSERT INTO embed_cache (fingerprint, text_hash, vector)
                 VALUES (?1, ?2, ?3)
                 ON CONFLICT(fingerprint, text_hash) DO UPDATE SET vector = excluded.vector",
            )?;
            for (text, vec) in entries {
                let hash = text_hash(text);
                stmt.execute(rusqlite::params![fingerprint, hash, f32_to_bytes(vec)])?;
            }
        }
        tx.commit().context("commit cache tx")?;
        Ok(())
    }

    /// Row count — used by tests + `search_status` diagnostics.
    pub fn len(&self) -> anyhow::Result<u64> {
        let conn = self.conn.lock().map_err(|e| anyhow!("cache mutex: {e}"))?;
        let n: i64 = conn
            .query_row("SELECT COUNT(*) FROM embed_cache", [], |r| r.get(0))
            .context("count embed_cache")?;
        Ok(n as u64)
    }
}

/// SHA-256 of the raw text bytes, hex-encoded. Stable across runs and
/// across machines; used verbatim as the cache key. Matches the
/// hashing strategy used in `Record::content_hash` so a future
/// optimization could keep a single hash across both layers.
pub fn text_hash(text: &str) -> String {
    hex::encode(Sha256::digest(text.as_bytes()))
}

fn f32_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for &x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn bytes_to_f32(b: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(b.len() / 4);
    for chunk in b.chunks_exact(4) {
        let word = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        out.push(f32::from_bits(word));
    }
    out
}

/// Decorator that memoizes the wrapped embedder's output in an
/// [`EmbedCache`]. The inner embedder is called only for misses;
/// dim + fingerprint are passed through unchanged so the decorator is
/// invisible to the indexer.
pub struct CachedEmbedder<E: Embedder> {
    inner: E,
    cache: EmbedCache,
}

impl<E: Embedder> CachedEmbedder<E> {
    pub fn new(inner: E, cache: EmbedCache) -> Self {
        Self { inner, cache }
    }

    pub fn cache(&self) -> &EmbedCache {
        &self.cache
    }
}

impl<E: Embedder> Embedder for CachedEmbedder<E> {
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn fingerprint(&self) -> String {
        self.inner.fingerprint()
    }

    fn embed(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let fp = self.inner.fingerprint();

        // Partition into (cached, to-embed) while preserving the
        // original input order for the returned Vec.
        let mut out: Vec<Option<Vec<f32>>> = vec![None; texts.len()];
        let mut miss_idx: Vec<usize> = Vec::new();
        let mut miss_texts: Vec<&str> = Vec::new();
        for (i, t) in texts.iter().enumerate() {
            match self.cache.get(&fp, t)? {
                Some(v) => out[i] = Some(v),
                None => {
                    miss_idx.push(i);
                    miss_texts.push(*t);
                }
            }
        }

        if !miss_texts.is_empty() {
            let fresh = self.inner.embed(&miss_texts)?;
            if fresh.len() != miss_texts.len() {
                return Err(anyhow!(
                    "inner embedder returned {} vectors for {} texts",
                    fresh.len(),
                    miss_texts.len()
                ));
            }

            // Collect entries for the batch write before moving
            // vectors into `out`. The loop below takes ownership.
            let entries: Vec<(&str, &[f32])> = miss_texts
                .iter()
                .zip(fresh.iter())
                .map(|(t, v)| (*t, v.as_slice()))
                .collect();
            self.cache.put_many(&fp, &entries)?;

            for (dst, vec) in miss_idx.iter().zip(fresh) {
                out[*dst] = Some(vec);
            }
        }

        // Every slot must be populated; `Option::unwrap` would panic
        // on a logic bug — surface it instead.
        let mut v = Vec::with_capacity(out.len());
        for (i, slot) in out.into_iter().enumerate() {
            v.push(slot.ok_or_else(|| anyhow!("cache partition left slot {i} empty"))?);
        }
        Ok(v)
    }

    fn embed_query(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        // Query-side embeddings get a different prefix ("Query: " for
        // Jina v5) and so produce different vectors than their
        // doc-side counterparts; we intentionally do NOT share the
        // cache with `embed`. The fingerprint captures doc vs. query
        // intent only at the embedder identity level, so we key the
        // cache with a suffix the embedder itself doesn't know about.
        let fp = format!("{}|query", self.inner.fingerprint());
        if let Some(v) = self.cache.get(&fp, text)? {
            return Ok(v);
        }
        let v = self.inner.embed_query(text)?;
        self.cache.put(&fp, text, &v)?;
        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::FakeEmbedder;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Wraps `FakeEmbedder` and counts how many times each method was
    /// called on the inner embedder — exactly what we need to prove
    /// the cache is short-circuiting calls on hit.
    struct CountingEmbedder {
        inner: FakeEmbedder,
        batch_calls: AtomicUsize,
        batch_texts: AtomicUsize,
        query_calls: AtomicUsize,
    }

    impl CountingEmbedder {
        fn new(dim: usize) -> Self {
            Self {
                inner: FakeEmbedder::new(dim),
                batch_calls: AtomicUsize::new(0),
                batch_texts: AtomicUsize::new(0),
                query_calls: AtomicUsize::new(0),
            }
        }
    }

    impl Embedder for CountingEmbedder {
        fn dim(&self) -> usize {
            self.inner.dim()
        }
        fn fingerprint(&self) -> String {
            self.inner.fingerprint()
        }
        fn embed(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
            self.batch_calls.fetch_add(1, Ordering::SeqCst);
            self.batch_texts
                .fetch_add(texts.len(), Ordering::SeqCst);
            self.inner.embed(texts)
        }
        fn embed_query(&self, text: &str) -> anyhow::Result<Vec<f32>> {
            self.query_calls.fetch_add(1, Ordering::SeqCst);
            self.inner.embed_query(text)
        }
    }

    #[test]
    fn cold_cache_forwards_all_texts_to_inner() {
        let cache = EmbedCache::in_memory().unwrap();
        let inner = CountingEmbedder::new(8);
        let cached = CachedEmbedder::new(inner, cache);

        let out = cached.embed(&["alpha", "beta"]).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(cached.inner.batch_texts.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn warm_cache_short_circuits_inner() {
        let cache = EmbedCache::in_memory().unwrap();
        let inner = CountingEmbedder::new(8);
        let cached = CachedEmbedder::new(inner, cache);

        cached.embed(&["alpha", "beta"]).unwrap();
        assert_eq!(cached.inner.batch_texts.load(Ordering::SeqCst), 2);

        // Same inputs again — cache should cover both.
        cached.embed(&["alpha", "beta"]).unwrap();
        assert_eq!(
            cached.inner.batch_texts.load(Ordering::SeqCst),
            2,
            "warm cache must not re-call the inner embedder"
        );
    }

    #[test]
    fn partial_hits_only_forward_misses() {
        let cache = EmbedCache::in_memory().unwrap();
        let inner = CountingEmbedder::new(8);
        let cached = CachedEmbedder::new(inner, cache);

        cached.embed(&["alpha"]).unwrap();
        assert_eq!(cached.inner.batch_texts.load(Ordering::SeqCst), 1);

        // Second batch: "alpha" is cached, "beta" is new → only one
        // text should go to the inner embedder.
        cached.embed(&["alpha", "beta"]).unwrap();
        assert_eq!(
            cached.inner.batch_texts.load(Ordering::SeqCst),
            2,
            "miss-only forwarding: alpha cached, beta new"
        );
    }

    #[test]
    fn cached_vectors_match_fresh_outputs() {
        let cache = EmbedCache::in_memory().unwrap();
        let inner = CountingEmbedder::new(16);
        let cached = CachedEmbedder::new(inner, cache);

        let first = cached.embed(&["hello"]).unwrap();
        let second = cached.embed(&["hello"]).unwrap();
        assert_eq!(first, second, "cache must return bit-identical vectors");
    }

    #[test]
    fn query_cache_is_separate_from_doc_cache() {
        // A fake embedder produces the same vector for a text under
        // both embed() and embed_query() (no prefix), so the cache
        // hit rate on the doc side should be zero when we've only
        // exercised the query side — confirming they're keyed apart.
        let cache = EmbedCache::in_memory().unwrap();
        let inner = CountingEmbedder::new(8);
        let cached = CachedEmbedder::new(inner, cache);

        cached.embed_query("hello").unwrap();
        assert_eq!(cached.inner.query_calls.load(Ordering::SeqCst), 1);

        // Same text on the doc side — this is a cold doc-side cache
        // and MUST hit the inner embedder.
        cached.embed(&["hello"]).unwrap();
        assert_eq!(cached.inner.batch_texts.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn different_fingerprints_dont_collide() {
        let cache = EmbedCache::in_memory().unwrap();
        // Same text, different fingerprints → different cache rows.
        cache.put("fp-a", "hello", &[1.0, 0.0]).unwrap();
        cache.put("fp-b", "hello", &[0.0, 1.0]).unwrap();
        assert_eq!(cache.get("fp-a", "hello").unwrap().unwrap(), vec![1.0, 0.0]);
        assert_eq!(cache.get("fp-b", "hello").unwrap().unwrap(), vec![0.0, 1.0]);
    }

    #[test]
    fn put_overwrites_on_conflict() {
        let cache = EmbedCache::in_memory().unwrap();
        cache.put("fp", "x", &[1.0]).unwrap();
        cache.put("fp", "x", &[2.0]).unwrap();
        assert_eq!(cache.get("fp", "x").unwrap().unwrap(), vec![2.0]);
        assert_eq!(cache.len().unwrap(), 1);
    }

    #[test]
    fn open_creates_parent_directory() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested").join("sub").join("cache.db");
        assert!(!path.parent().unwrap().exists());
        let c = EmbedCache::open(&path).unwrap();
        assert!(path.parent().unwrap().exists());
        c.put("fp", "x", &[0.1, 0.2]).unwrap();
        drop(c);

        // Reopening sees the persisted row.
        let c2 = EmbedCache::open(&path).unwrap();
        assert_eq!(c2.get("fp", "x").unwrap().unwrap(), vec![0.1, 0.2]);
    }
}
