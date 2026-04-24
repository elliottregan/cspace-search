//! sqlite-vec backed [`Upserter`] implementation.
//!
//! One sqlite file per project holds every corpus as its own vec0
//! virtual table. Auxiliary columns (`+content_hash`, `+payload`) live
//! in the same table so change detection and payload lookup don't need
//! a join. Rowids are the 64-bit [`Point::id`] cast to `i64` — sqlite
//! stores them natively and the cast is a no-op on the bit pattern.

use super::{Point, RawHit, Searcher, Upserter};
use anyhow::{anyhow, Context};
use rusqlite::Connection;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, Once};

/// Register sqlite-vec as an auto-loaded extension the first time any
/// [`SqliteUpserter`] is opened. Every subsequent `Connection::open`
/// in the process picks it up automatically. Safe to call repeatedly.
fn ensure_vec_extension_registered() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        // Safety: the documented registration entry point for
        // sqlite-vec. `sqlite3_auto_extension` takes a function pointer
        // of shape `int (*)(sqlite3*, char**, sqlite3_api_routines*)`;
        // we transmute the sqlite-vec init symbol into that shape.
        // sqlite-vec's signature differs only in the first argument's
        // crate path, and calls through both are ABI-identical.
        type AutoExt = unsafe extern "C" fn(
            *mut rusqlite::ffi::sqlite3,
            *mut *mut std::os::raw::c_char,
            *const rusqlite::ffi::sqlite3_api_routines,
        ) -> std::os::raw::c_int;
        unsafe {
            rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute::<*const (), AutoExt>(
                sqlite_vec::sqlite3_vec_init as *const (),
            )));
        }
    });
}

pub struct SqliteUpserter {
    path: PathBuf,
    conn: Mutex<Connection>,
}

impl SqliteUpserter {
    /// Open (or create) the sqlite file at `path`. Parent directories
    /// are created as needed.
    pub fn open(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        ensure_vec_extension_registered();
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).context("creating sqlite parent dir")?;
        }
        let conn = Connection::open(&path).context("opening sqlite db")?;
        // WAL mode is kinder to concurrent readers (the MCP server can
        // query while `init` is upserting).
        conn.pragma_update(None, "journal_mode", "WAL")
            .context("enabling WAL mode")?;
        Ok(Self {
            path,
            conn: Mutex::new(conn),
        })
    }

    /// Open a fresh in-memory instance. For tests.
    pub fn in_memory() -> anyhow::Result<Self> {
        ensure_vec_extension_registered();
        let conn = Connection::open_in_memory().context("opening in-memory sqlite")?;
        Ok(Self {
            path: PathBuf::new(),
            conn: Mutex::new(conn),
        })
    }
}

fn quote_ident(name: &str) -> String {
    // sqlite identifier quoting: double-quote and escape any embedded
    // double-quotes by doubling. Our collection names are
    // `<corpus>-<hash>` — hyphens require quoting to be treated as a
    // single identifier.
    format!("\"{}\"", name.replace('"', "\"\""))
}

/// Convert `[f32]` to the little-endian byte slice sqlite-vec expects.
fn vec_blob(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for &x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

impl Upserter for SqliteUpserter {
    fn ensure_collection(&self, name: &str, dim: usize) -> anyhow::Result<()> {
        let conn = self.conn.lock().map_err(|e| anyhow!("sqlite mutex: {e}"))?;
        let quoted = quote_ident(name);
        // vec0 virtual tables support auxiliary columns prefixed with `+`.
        // They're stored as regular sqlite values and aren't indexed for
        // vector search, which is exactly what we want for the payload.
        let sql = format!(
            "CREATE VIRTUAL TABLE IF NOT EXISTS {quoted} USING vec0(
                embedding float[{dim}],
                +content_hash TEXT,
                +payload TEXT
            )"
        );
        conn.execute(&sql, [])
            .with_context(|| format!("ensure_collection({name})"))?;
        Ok(())
    }

    fn upsert_points(
        &self,
        collection: &str,
        points: &[Point],
        batch_size: usize,
        progress: Option<&dyn Fn(usize, usize)>,
    ) -> anyhow::Result<()> {
        if points.is_empty() {
            return Ok(());
        }
        let mut conn = self.conn.lock().map_err(|e| anyhow!("sqlite mutex: {e}"))?;
        let quoted = quote_ident(collection);
        let total = points.len();
        let batch_size = batch_size.max(1);

        let mut offset = 0;
        while offset < total {
            let end = (offset + batch_size).min(total);
            let tx = conn
                .transaction()
                .with_context(|| format!("begin upsert tx for {collection}"))?;

            // Delete first so we get INSERT-OR-REPLACE semantics without
            // running into vec0's quirk around direct REPLACE on virtual
            // tables — cleanest is delete-then-insert in one transaction.
            {
                let del_sql = format!("DELETE FROM {quoted} WHERE rowid = ?1");
                let mut del = tx.prepare_cached(&del_sql)?;
                for p in &points[offset..end] {
                    del.execute([p.id as i64])?;
                }
            }
            {
                let ins_sql = format!(
                    "INSERT INTO {quoted} (rowid, embedding, content_hash, payload)
                     VALUES (?1, ?2, ?3, ?4)"
                );
                let mut ins = tx.prepare_cached(&ins_sql)?;
                for p in &points[offset..end] {
                    let hash = p
                        .payload
                        .get("content_hash")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let payload_json =
                        serde_json::to_string(&p.payload).context("serializing point payload")?;
                    ins.execute(rusqlite::params![
                        p.id as i64,
                        vec_blob(&p.vector),
                        hash,
                        payload_json,
                    ])?;
                }
            }
            tx.commit().context("commit upsert tx")?;

            if let Some(cb) = progress {
                cb(end, total);
            }
            offset = end;
        }
        Ok(())
    }

    fn existing_points(&self, collection: &str) -> anyhow::Result<BTreeMap<u64, String>> {
        let conn = self.conn.lock().map_err(|e| anyhow!("sqlite mutex: {e}"))?;
        let quoted = quote_ident(collection);
        let sql = format!("SELECT rowid, content_hash FROM {quoted}");
        // Missing collections are a valid state (fresh project, no
        // `init` yet) — surface as "nothing exists", not as an error.
        let mut stmt = match conn.prepare(&sql) {
            Ok(s) => s,
            Err(rusqlite::Error::SqliteFailure(_, Some(ref msg)))
                if msg.contains("no such table") =>
            {
                return Ok(BTreeMap::new());
            }
            Err(e) => return Err(anyhow::Error::from(e)).context("prepare existing_points"),
        };
        let rows = stmt.query_map([], |r| {
            let id: i64 = r.get(0)?;
            let hash: Option<String> = r.get(1)?;
            Ok((id as u64, hash.unwrap_or_default()))
        })?;
        let mut out = BTreeMap::new();
        for row in rows {
            let (id, hash) = row?;
            out.insert(id, hash);
        }
        Ok(out)
    }

    fn delete_points(&self, collection: &str, ids: &[u64]) -> anyhow::Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let mut conn = self.conn.lock().map_err(|e| anyhow!("sqlite mutex: {e}"))?;
        let quoted = quote_ident(collection);
        let tx = conn.transaction()?;
        {
            let sql = format!("DELETE FROM {quoted} WHERE rowid = ?1");
            let mut stmt = tx.prepare_cached(&sql)?;
            for &id in ids {
                stmt.execute([id as i64])?;
            }
        }
        tx.commit().context("commit delete tx")?;
        Ok(())
    }
}

impl Searcher for SqliteUpserter {
    fn search(
        &self,
        collection: &str,
        vector: &[f32],
        top_k: usize,
    ) -> anyhow::Result<Vec<RawHit>> {
        if top_k == 0 {
            return Ok(Vec::new());
        }
        let conn = self.conn.lock().map_err(|e| anyhow!("sqlite mutex: {e}"))?;
        let quoted = quote_ident(collection);
        let sql = format!(
            "SELECT rowid, distance, payload FROM {quoted} \
             WHERE embedding MATCH ?1 ORDER BY distance LIMIT ?2"
        );
        let mut stmt = match conn.prepare(&sql) {
            Ok(s) => s,
            Err(rusqlite::Error::SqliteFailure(_, Some(ref msg)))
                if msg.contains("no such table") =>
            {
                return Ok(Vec::new());
            }
            Err(e) => return Err(anyhow::Error::from(e)).context("prepare search"),
        };
        let limit = i64::try_from(top_k).unwrap_or(i64::MAX);
        let q_blob = vec_blob(vector);
        let rows = stmt.query_map(rusqlite::params![q_blob, limit], |r| {
            let id: i64 = r.get(0)?;
            let dist: f64 = r.get(1)?;
            let payload_json: Option<String> = r.get(2)?;
            Ok((id as u64, dist as f32, payload_json))
        })?;

        let mut out = Vec::new();
        for row in rows {
            let (id, distance, payload_json) = row?;
            // L2 distance → similarity score on [0, 1]. Monotonic
            // decreasing in distance; cosine order is preserved since
            // stored vectors are L2-normalized.
            let score = 1.0f32 / (1.0 + distance);
            let payload: BTreeMap<String, serde_json::Value> = match payload_json {
                Some(s) => serde_json::from_str(&s).unwrap_or_default(),
                None => BTreeMap::new(),
            };
            out.push(RawHit { id, score, payload });
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn mk_point(id: u64, hash: &str, dim: usize) -> Point {
        let mut payload = BTreeMap::new();
        payload.insert("content_hash".into(), json!(hash));
        payload.insert("path".into(), json!(format!("file-{id}.rs")));
        Point {
            id,
            vector: vec![0.1; dim],
            payload,
        }
    }

    #[test]
    fn ensure_collection_is_idempotent() {
        let u = SqliteUpserter::in_memory().unwrap();
        u.ensure_collection("commits-abc", 8).unwrap();
        u.ensure_collection("commits-abc", 8).unwrap();
    }

    #[test]
    fn upsert_then_existing_points_round_trips() {
        let u = SqliteUpserter::in_memory().unwrap();
        u.ensure_collection("code-xyz", 4).unwrap();
        let pts = vec![
            mk_point(1, "aaa", 4),
            mk_point(2, "bbb", 4),
            mk_point(u64::MAX, "ccc", 4), // exercises the i64 bit-cast.
        ];
        u.upsert_points("code-xyz", &pts, 2, None).unwrap();

        let existing = u.existing_points("code-xyz").unwrap();
        assert_eq!(existing.len(), 3);
        assert_eq!(existing.get(&1).unwrap(), "aaa");
        assert_eq!(existing.get(&2).unwrap(), "bbb");
        assert_eq!(existing.get(&u64::MAX).unwrap(), "ccc");
    }

    #[test]
    fn upsert_replaces_by_id() {
        let u = SqliteUpserter::in_memory().unwrap();
        u.ensure_collection("code", 4).unwrap();
        u.upsert_points("code", &[mk_point(7, "old", 4)], 8, None)
            .unwrap();
        u.upsert_points("code", &[mk_point(7, "new", 4)], 8, None)
            .unwrap();
        let existing = u.existing_points("code").unwrap();
        assert_eq!(existing.len(), 1);
        assert_eq!(existing.get(&7).unwrap(), "new");
    }

    #[test]
    fn existing_points_on_missing_collection_returns_empty() {
        let u = SqliteUpserter::in_memory().unwrap();
        let existing = u.existing_points("never-created").unwrap();
        assert!(existing.is_empty());
    }

    #[test]
    fn delete_points_removes_selected_ids() {
        let u = SqliteUpserter::in_memory().unwrap();
        u.ensure_collection("code", 4).unwrap();
        u.upsert_points(
            "code",
            &[
                mk_point(1, "a", 4),
                mk_point(2, "b", 4),
                mk_point(3, "c", 4),
            ],
            8,
            None,
        )
        .unwrap();
        u.delete_points("code", &[1, 3]).unwrap();
        let existing = u.existing_points("code").unwrap();
        assert_eq!(existing.len(), 1);
        assert!(existing.contains_key(&2));
    }

    #[test]
    fn upsert_reports_progress_per_batch() {
        let u = SqliteUpserter::in_memory().unwrap();
        u.ensure_collection("code", 4).unwrap();
        let pts: Vec<Point> = (1..=10).map(|i| mk_point(i, "h", 4)).collect();

        let calls = std::cell::RefCell::new(Vec::<(usize, usize)>::new());
        let cb = |done, total| calls.borrow_mut().push((done, total));
        u.upsert_points("code", &pts, 3, Some(&cb)).unwrap();

        // Ten points at batch 3 → batches of [3, 3, 3, 1] → progress
        // reports at done = 3, 6, 9, 10.
        let observed = calls.into_inner();
        assert_eq!(
            observed,
            vec![(3, 10), (6, 10), (9, 10), (10, 10)],
            "progress callback should fire once per batch"
        );
    }

    #[test]
    fn knn_search_returns_nearest_first() {
        // Basic sanity that vec0 actually does what we expect: three
        // orthogonal unit vectors, query with one of them, the matching
        // row should rank first.
        let u = SqliteUpserter::in_memory().unwrap();
        u.ensure_collection("code", 3).unwrap();
        let pts = vec![
            Point {
                id: 1,
                vector: vec![1.0, 0.0, 0.0],
                payload: [("content_hash".into(), json!("a"))].into_iter().collect(),
            },
            Point {
                id: 2,
                vector: vec![0.0, 1.0, 0.0],
                payload: [("content_hash".into(), json!("b"))].into_iter().collect(),
            },
            Point {
                id: 3,
                vector: vec![0.0, 0.0, 1.0],
                payload: [("content_hash".into(), json!("c"))].into_iter().collect(),
            },
        ];
        u.upsert_points("code", &pts, 8, None).unwrap();

        let conn = u.conn.lock().unwrap();
        let query: Vec<u8> = vec_blob(&[0.9, 0.1, 0.0]); // closest to id=1
        let mut stmt = conn
            .prepare(
                "SELECT rowid, distance FROM \"code\" WHERE embedding MATCH ?1 \
                 ORDER BY distance LIMIT 3",
            )
            .unwrap();
        let rows: Vec<(i64, f64)> = stmt
            .query_map([query], |r| Ok((r.get(0)?, r.get(1)?)))
            .unwrap()
            .map(Result::unwrap)
            .collect();
        assert_eq!(rows.len(), 3, "expected three hits");
        assert_eq!(rows[0].0, 1, "id=1 should rank first, got {:?}", rows);
    }
}
