//! Config loading: deep-merges a project's `search.yaml` over the embedded
//! default. Ported from the Go `search/config` package so YAML schemas
//! remain identical and existing `search.yaml` files work unchanged.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

pub mod runtime;
pub use runtime::resolve_corpus;

/// Embedded default configuration. Must stay in sync with the Go version
/// at `cspace/search/config/default.yaml` until cspace drops the search
/// subcommand entirely.
const DEFAULT_YAML: &str = include_str!("default.yaml");

/// Top-level config shape. Mirrors `search/config/config.go:Config`.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Config {
    /// Master switch for semantic search in this project. Must be set
    /// to true explicitly in `search.yaml`; otherwise every search CLI,
    /// MCP, and bootstrap path fast-exits with an error.
    ///
    /// Prevents "I just cloned a repo with cspace baked in and now it's
    /// indexing node_modules on first container boot."
    #[serde(default)]
    pub enabled: bool,

    #[serde(default)]
    pub corpora: BTreeMap<String, CorpusConfig>,

    #[serde(default)]
    pub sidecars: Sidecars,

    #[serde(default)]
    pub index: IndexConfig,
}

/// Per-corpus configuration. The shape is a union of every field any
/// corpus builder cares about — unused fields are ignored by the
/// factory, and every field has a sensible default. Explicit over
/// tagged-union-deserialize since the YAML deep-merge layer operates
/// on `serde_yaml::Value` before this struct ever gets constructed.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct CorpusConfig {
    #[serde(default)]
    pub enabled: bool,

    // --- shared across file-style corpora ---
    /// Reject files larger than this many bytes (FileCorpus). Zero or
    /// absent means no size cap.
    #[serde(default)]
    pub max_bytes: i64,
    /// Glob excludes applied after source enumeration (FileCorpus).
    #[serde(default)]
    pub excludes: Vec<String>,

    // --- CommitCorpus ---
    /// Max commits to index. Zero falls back to the CommitCorpus
    /// default (500).
    #[serde(default)]
    pub limit: i64,

    // --- FileCorpus ---
    /// Corpus builder dispatch key. Known: `files`, `commits`. If
    /// absent, the built-in default for `id` wins.
    #[serde(default, rename = "type")]
    pub type_name: Option<String>,

    /// Where to enumerate files from. One of `git-ls-files`,
    /// `filesystem`, `walk`. Only meaningful when `type = files`.
    #[serde(default)]
    pub source: Option<String>,

    /// Chunking at the corpus level. Every PathGroup inherits this
    /// unless it supplies its own override. Absent means no chunking
    /// (files go through as a single record, truncated only by the
    /// embed-text budget).
    #[serde(default)]
    pub chunk: Option<ChunkSpec>,

    /// Explicit path groups. When absent, FileCorpus falls back to a
    /// single group matching everything the `source` enumerates.
    #[serde(default)]
    pub path_groups: Vec<PathGroupSpec>,

    /// Header prepended to every Record's `embed_text`. Supports
    /// `{path}`, `{kind}`, `{basename}`, `{basename_no_ext}`, plus
    /// any key from a path group's `extra` map. Include any trailing
    /// blank lines you want explicitly — the engine does not append
    /// `\n\n` for you.
    #[serde(default)]
    pub embed_header: Option<String>,

    /// Default `kind` for records whose PathGroup doesn't override.
    /// Multi-chunk records override to `"chunk"` per the Go convention.
    #[serde(default)]
    pub record_kind: Option<String>,

    /// Truncate each `embed_text` to this many chars after header +
    /// body concatenation. Falls back to 12 000.
    #[serde(default)]
    pub max_embed_chars: Option<usize>,
}

/// Chunking spec, matches the existing `corpus::chunker::ChunkConfig`
/// shape. Kept as a separate serde type to keep `ChunkConfig` a pure
/// runtime concern.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ChunkSpec {
    #[serde(default)]
    pub max: usize,
    #[serde(default)]
    pub overlap: usize,
}

/// One include/kind group within a FileCorpus.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct PathGroupSpec {
    #[serde(default)]
    pub include: Vec<String>,
    #[serde(default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub chunk: Option<ChunkSpec>,
    /// Extra fields to attach to every Record this group produces.
    /// Values are templates — the same `{path}` / `{basename_no_ext}`
    /// / etc. substitutions as `embed_header`. Evaluated after the
    /// built-in vars so a key named `path` would overwrite.
    #[serde(default)]
    pub extra: BTreeMap<String, String>,
}

/// External service URLs. Preserved for transitional compatibility while
/// the Rust port still routes through llama-server / qdrant / reduce-api
/// over HTTP. Later phases swap most of these for in-process
/// implementations; see PORTING.md.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Sidecars {
    #[serde(default)]
    pub llama_retrieval_url: String,
    #[serde(default)]
    pub llama_clustering_url: String,
    #[serde(default)]
    pub qdrant_url: String,
    #[serde(default)]
    pub reduce_url: String,
    #[serde(default)]
    pub hdbscan_url: String,
}

/// Indexer runtime paths (lock file, log file), interpreted relative to
/// the project root.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct IndexConfig {
    #[serde(default)]
    pub lock_path: String,
    #[serde(default)]
    pub log_path: String,
}

/// Load the embedded defaults and deep-merge an optional `search.yaml`
/// from `project_root` on top.
///
/// Deep-merge matters for nested maps like `corpora.<id>`: a naive
/// overwrite would wipe defaulted fields every time the user sets any
/// single leaf (e.g. overriding `corpora.code.excludes` would reset
/// `max_bytes` to zero). Here we merge at the parsed-YAML tree level and
/// then deserialize the merged tree, so only the keys actually present
/// in the overlay take effect.
pub fn load(project_root: impl AsRef<Path>) -> anyhow::Result<Config> {
    let mut base: serde_yaml::Value = serde_yaml::from_str(DEFAULT_YAML)?;
    let override_path = project_root.as_ref().join("search.yaml");
    match std::fs::read_to_string(&override_path) {
        Ok(text) => {
            let overlay: serde_yaml::Value = serde_yaml::from_str(&text)?;
            deep_merge(&mut base, overlay);
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e.into()),
    }
    Ok(serde_yaml::from_value(base)?)
}

/// Recursively merge `overlay` into `base`: nested maps merge key by key
/// (overlay wins on leaf conflicts); scalars and sequences from the
/// overlay replace base values wholesale. Ports `search/config/config.go:deepMerge`.
fn deep_merge(base: &mut serde_yaml::Value, overlay: serde_yaml::Value) {
    use serde_yaml::Value;
    match (base, overlay) {
        (Value::Mapping(bm), Value::Mapping(om)) => {
            for (k, v) in om {
                match bm.get_mut(&k) {
                    Some(bv) => deep_merge(bv, v),
                    None => {
                        bm.insert(k, v);
                    }
                }
            }
        }
        (b, o) => *b = o,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let c = load(dir.path()).unwrap();

        // Master switch defaults OFF so fresh cspace projects don't
        // auto-index node_modules on first bootstrap.
        assert!(!c.enabled);

        assert_eq!(c.corpora["code"].max_bytes, 204_800);
        assert!(!c.sidecars.qdrant_url.is_empty());
        assert!(!c.index.lock_path.is_empty());

        assert!(c.corpora["code"].enabled);
        assert!(c.corpora["commits"].enabled);
        assert!(!c.corpora["context"].enabled);
        assert!(!c.corpora["issues"].enabled);

        assert!(!c.corpora["code"].excludes.is_empty());
    }

    #[test]
    fn load_project_override() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("search.yaml"),
            "corpora:\n  code:\n    max_bytes: 50000\n    enabled: false\n",
        )
        .unwrap();

        let c = load(dir.path()).unwrap();
        assert_eq!(c.corpora["code"].max_bytes, 50_000);
        assert!(!c.corpora["code"].enabled);
        // Non-overridden fields still come from defaults.
        assert!(!c.sidecars.qdrant_url.is_empty());
    }

    #[test]
    fn load_master_switch_opt_in() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("search.yaml"), "enabled: true\n").unwrap();

        let c = load(dir.path()).unwrap();
        assert!(c.enabled);
        assert!(c.corpora["code"].enabled);
        assert!(!c.corpora["issues"].enabled);
    }

    #[test]
    fn load_opt_in_disabled_corpus() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("search.yaml"),
            "corpora:\n  issues:\n    enabled: true\n",
        )
        .unwrap();

        let c = load(dir.path()).unwrap();
        assert!(c.corpora["issues"].enabled);
        // Defaulted limit (500) survives partial override.
        assert_eq!(c.corpora["issues"].limit, 500);
    }

    #[test]
    fn deep_merge_preserves_sibling_keys() {
        use serde_yaml::Value;
        let mut base: Value = serde_yaml::from_str("a:\n  x: 1\n  y: 2\n").unwrap();
        let overlay: Value = serde_yaml::from_str("a:\n  x: 99\n").unwrap();
        deep_merge(&mut base, overlay);
        assert_eq!(base["a"]["x"], Value::from(99));
        assert_eq!(base["a"]["y"], Value::from(2));
    }

    #[test]
    fn deep_merge_scalar_replaces_map() {
        use serde_yaml::Value;
        let mut base: Value = serde_yaml::from_str("a: {x: 1}\n").unwrap();
        let overlay: Value = serde_yaml::from_str("a: 42\n").unwrap();
        deep_merge(&mut base, overlay);
        assert_eq!(base["a"], Value::from(42));
    }
}
