//! Config loading: deep-merges a project's `search.yaml` over the embedded
//! default. Ported from the Go `search/config` package so YAML schemas
//! remain identical and existing `search.yaml` files work unchanged.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

mod runtime;
pub use runtime::{resolve_corpus, SearchError};

/// Embedded default configuration. Must stay in sync with the Go version
/// at `cspace/search/config/default.yaml` until cspace drops the search
/// subcommand entirely.
const DEFAULT_YAML: &str = include_str!("default.yaml");

/// Top-level config shape. Mirrors `search/config/config.go:Config`.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Config {
    /// Master switch for semantic search in this project. Must be set to
    /// true explicitly in `search.yaml`; otherwise every search CLI + MCP
    /// + bootstrap path fast-exits with `SearchError::SearchDisabled`.
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

/// Per-corpus configuration.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct CorpusConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub max_bytes: i64,
    #[serde(default)]
    pub excludes: Vec<String>,
    #[serde(default)]
    pub limit: i64,
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
