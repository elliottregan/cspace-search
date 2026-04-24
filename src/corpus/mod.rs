//! Corpus: the abstraction over indexable content types.
//!
//! A [`Corpus`] enumerates [`Record`]s; later phases embed them and write
//! them to the vector store. The same pipeline serves commits, code,
//! context, and issues. Port of the Go `search/corpus` package.

use serde::Serialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

mod commits;
pub use commits::CommitCorpus;

/// One unit of content to be indexed. May represent a whole file, a
/// chunk of a file, or a commit; [`Record::kind`] disambiguates.
#[derive(Debug, Clone, Serialize)]
pub struct Record {
    /// Primary identifier: relative file path (code), commit hash
    /// (commits), or similar.
    pub path: String,

    /// 1-based inclusive line range for chunked code records. Zero for
    /// whole-file or commit records.
    pub line_start: u32,
    pub line_end: u32,

    /// One of `"file"`, `"chunk"`, `"commit"`.
    pub kind: String,

    /// Hex sha256 of the source bytes, used for change detection.
    /// Empty for corpora that don't checksum content (commits).
    pub content_hash: String,

    /// Corpus-specific metadata that should land in the vector-store
    /// payload (e.g. commit subject + date).
    pub extra: BTreeMap<String, serde_json::Value>,

    /// Text sent to the embedding model. Populated by the corpus at
    /// enumeration time so the indexer does not need to know the
    /// corpus-specific embedding format.
    pub embed_text: String,
}

impl Record {
    /// Deterministic 64-bit point ID for this record. Ported verbatim
    /// from the Go version's hash layout so records keyed by the Go
    /// implementation continue to round-trip.
    pub fn id(&self) -> u64 {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(self.kind.as_bytes());
        h.update([0]);
        h.update(self.path.as_bytes());
        h.update([0]);
        h.update(self.line_start.to_string().as_bytes());
        h.update([0]);
        h.update(self.line_end.to_string().as_bytes());
        let sum = h.finalize();
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&sum[..8]);
        u64::from_be_bytes(bytes)
    }
}

/// Stable 8-char hex hash of a project root's absolute path, used as a
/// collection-name suffix.
pub fn project_hash(repo_path: &Path) -> String {
    use sha2::{Digest, Sha256};
    let abs: PathBuf = if repo_path.is_absolute() {
        repo_path.to_path_buf()
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(repo_path))
            .unwrap_or_else(|_| repo_path.to_path_buf())
    };
    let mut h = Sha256::new();
    h.update(abs.to_string_lossy().as_bytes());
    let sum = h.finalize();
    hex::encode(&sum[..4])
}

/// Truncate a `&str` to at most `max_bytes`, backing up to the nearest
/// UTF-8 character boundary so the returned slice is always valid.
pub(crate) fn truncate_utf8(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// An indexable content type (commits, code, context, issues).
pub trait Corpus {
    /// Stable corpus identifier (`"commits"`, `"code"`, etc).
    fn id(&self) -> &'static str;

    /// Vector-store collection name for this corpus + project.
    fn collection(&self, project_root: &Path) -> String;

    /// Emit all records for this corpus. The Go version returned two
    /// channels (records + errs); Rust prefers `Result<Vec<_>>` since
    /// every corpus here produces a bounded number of records and the
    /// indexer batches them anyway. Can swap to a streaming iterator if
    /// a future corpus blows past that assumption.
    fn enumerate(&self, project_root: &Path) -> anyhow::Result<Vec<Record>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_id_stable_for_same_inputs() {
        let r1 = Record {
            path: "foo.go".into(),
            line_start: 10,
            line_end: 20,
            kind: "chunk".into(),
            content_hash: String::new(),
            extra: BTreeMap::new(),
            embed_text: String::new(),
        };
        let r2 = Record {
            path: "foo.go".into(),
            line_start: 10,
            line_end: 20,
            kind: "chunk".into(),
            content_hash: "different".into(),
            extra: {
                let mut m = BTreeMap::new();
                m.insert("noise".into(), serde_json::Value::Bool(true));
                m
            },
            embed_text: "also different".into(),
        };
        assert_eq!(
            r1.id(),
            r2.id(),
            "Record::id depends only on kind+path+lines"
        );
    }

    #[test]
    fn record_id_differs_on_kind_or_range() {
        let base = Record {
            path: "foo.go".into(),
            line_start: 10,
            line_end: 20,
            kind: "chunk".into(),
            content_hash: String::new(),
            extra: BTreeMap::new(),
            embed_text: String::new(),
        };
        let variants = [
            (
                "path",
                Record {
                    path: "bar.go".into(),
                    ..base.clone()
                },
            ),
            (
                "line_start",
                Record {
                    line_start: 1,
                    ..base.clone()
                },
            ),
            (
                "line_end",
                Record {
                    line_end: 21,
                    ..base.clone()
                },
            ),
            (
                "kind",
                Record {
                    kind: "file".into(),
                    ..base.clone()
                },
            ),
        ];
        for (name, v) in &variants {
            assert_ne!(v.id(), base.id(), "{name} should produce a different id");
        }
    }

    #[test]
    fn project_hash_is_stable_8_hex_chars() {
        let h = project_hash(Path::new("/tmp/example-project"));
        assert_eq!(h.len(), 8, "expected 8-char hex, got {h}");
        assert!(
            h.chars().all(|c| c.is_ascii_hexdigit()),
            "non-hex char in {h}"
        );
        let h2 = project_hash(Path::new("/tmp/example-project"));
        assert_eq!(h, h2);
    }

    #[test]
    fn truncate_utf8_backs_off_from_multibyte_split() {
        // "é" is two bytes in UTF-8. Truncating to 1 byte must back up to 0.
        assert_eq!(truncate_utf8("é", 1), "");
        assert_eq!(truncate_utf8("é", 2), "é");
        assert_eq!(truncate_utf8("abcé", 3), "abc");
        assert_eq!(truncate_utf8("abc", 10), "abc");
    }
}
