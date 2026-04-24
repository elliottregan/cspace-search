//! Single-chokepoint gating for `search`'s "is this corpus runnable?"
//! question. Every CLI query, init loop, and MCP tool routes through
//! [`resolve_corpus`] (gate only) or [`build`] / [`build_with_config`]
//! (gate + corpus factory) so a disabled search or disabled corpus
//! produces one consistent, actionable error.
//!
//! Ported from `search/config/runtime.go`.

// Phase 2b-5 wires Runtime + factory up through the tests; Phase 3
// (sqlite-vec storage) wires them into `init` and `search` subcommands.
// Until then, the non-test build path doesn't call build() yet.
#![allow(dead_code)]

use super::{Config, CorpusConfig};
use crate::corpus::{Corpus, CorpusRegistry};
use std::path::Path;
use std::sync::OnceLock;
use thiserror::Error;

/// Process-wide registry. Built lazily on first call so tests and
/// consumers that never touch the factory don't pay the init cost.
fn default_registry() -> &'static CorpusRegistry {
    static REG: OnceLock<CorpusRegistry> = OnceLock::new();
    REG.get_or_init(CorpusRegistry::with_builtins)
}

#[derive(Debug, Error)]
pub enum SearchError {
    #[error("search not configured for this project (set `enabled: true` in {0}/search.yaml to activate)")]
    SearchDisabled(String),

    #[error("{corpus:?}: corpus disabled in search.yaml (set corpora.{corpus}.enabled=true in search.yaml to enable)")]
    CorpusDisabled { corpus: String },

    #[error("unknown corpus {corpus:?} (known: code, commits, context, issues)")]
    UnknownCorpus { corpus: String },

    #[error("building corpus {corpus:?}: {detail}")]
    BuildFailed { corpus: String, detail: String },
}

/// Pairs a loaded config with a ready-to-run corpus for one command
/// invocation. Ported from Go's `corpus.Runtime`.
#[derive(Debug)]
pub struct Runtime {
    pub cfg: Config,
    pub corpus: Box<dyn Corpus>,
}

/// Load config from `project_root` and build the requested corpus.
/// Single-chokepoint for every CLI query, init loop, and MCP tool.
pub fn build(project_root: &Path, corpus_id: &str) -> anyhow::Result<Runtime> {
    let cfg = super::load(project_root)?;
    build_with_config(project_root, corpus_id, cfg)
}

/// Same as [`build`] but reuses an already-loaded config (avoids an
/// extra disk read when the caller already holds one).
pub fn build_with_config(
    project_root: &Path,
    corpus_id: &str,
    cfg: Config,
) -> anyhow::Result<Runtime> {
    // Gating first — master switch + per-corpus enable/disable.
    resolve_corpus(&cfg, corpus_id, project_root)?;
    let corpus = build_corpus(corpus_id, &cfg)?;
    Ok(Runtime { cfg, corpus })
}

/// Instantiate the named corpus with config-provided knobs.
///
/// Dispatch goes through the process-wide `CorpusRegistry`. The type
/// name comes from `corpora.<id>.type`, falling back to `"commits"`
/// for the `commits` id (for configs predating the `type` field) and
/// `"files"` otherwise. Unknown type names surface as `UnknownCorpus`
/// with the list of registered types for user diagnosis.
fn build_corpus(id: &str, cfg: &Config) -> Result<Box<dyn Corpus>, SearchError> {
    let cc = cfg.corpora.get(id).cloned().unwrap_or_default();
    let ty = match cc.type_name.as_deref() {
        Some(t) => t,
        None if id == "commits" => "commits",
        None => "files",
    };
    let reg = default_registry();
    match reg.build(ty, id, &cc) {
        Some(Ok(corpus)) => Ok(corpus),
        Some(Err(e)) => Err(SearchError::BuildFailed {
            corpus: id.to_string(),
            detail: e.to_string(),
        }),
        None => Err(SearchError::UnknownCorpus {
            corpus: format!(
                "{id} (type = {ty}; known: {})",
                reg.known_types().join(", ")
            ),
        }),
    }
}

/// Returns the `CorpusConfig` for `corpus_id`, or an error if the master
/// switch is off, the corpus is opted out, or the id isn't recognized.
///
/// Master-switch check supersedes per-corpus: if search is disabled at
/// the project level, no individual corpus is evaluated. Keeps fresh
/// cspace projects from incidentally indexing node_modules on first boot.
pub fn resolve_corpus<'a>(
    cfg: &'a Config,
    corpus_id: &str,
    project_root: &Path,
) -> Result<&'a CorpusConfig, SearchError> {
    if !cfg.enabled {
        return Err(SearchError::SearchDisabled(
            project_root.to_string_lossy().into_owned(),
        ));
    }
    match cfg.corpora.get(corpus_id) {
        Some(cc) if !cc.enabled => Err(SearchError::CorpusDisabled {
            corpus: corpus_id.to_string(),
        }),
        Some(cc) => Ok(cc),
        None => Err(SearchError::UnknownCorpus {
            corpus: corpus_id.to_string(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config;

    /// With the master switch off (default), no corpus resolves.
    /// Single-point gate.
    #[test]
    fn search_disabled_blocks_everything() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = config::load(dir.path()).unwrap();
        for id in ["code", "commits", "context", "issues"] {
            let err = resolve_corpus(&cfg, id, dir.path()).unwrap_err();
            assert!(
                matches!(err, SearchError::SearchDisabled(_)),
                "{id}: expected SearchDisabled, got {err:?}"
            );
        }
    }

    /// Once the master switch is on, per-corpus enable/disable takes over.
    /// code+commits resolve; context+issues return CorpusDisabled.
    #[test]
    fn per_corpus_after_master_switch() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("search.yaml"), "enabled: true\n").unwrap();
        let cfg = config::load(dir.path()).unwrap();

        for id in ["code", "commits"] {
            resolve_corpus(&cfg, id, dir.path())
                .unwrap_or_else(|e| panic!("{id} expected Ok, got {e:?}"));
        }
        for id in ["context", "issues"] {
            let err = resolve_corpus(&cfg, id, dir.path()).unwrap_err();
            assert!(
                matches!(err, SearchError::CorpusDisabled { .. }),
                "{id}: expected CorpusDisabled, got {err:?}"
            );
        }
    }

    #[test]
    fn unknown_corpus_reports_known_ids() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("search.yaml"), "enabled: true\n").unwrap();
        let cfg = config::load(dir.path()).unwrap();

        let err = resolve_corpus(&cfg, "bogus", dir.path()).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("bogus"), "error omitted bogus id: {msg}");
        assert!(msg.contains("commits"), "error omitted known list: {msg}");
    }

    /// `build_with_config` instantiates a Runtime with the right corpus
    /// trait object when the gate is open.
    #[test]
    fn build_with_config_returns_matching_corpus() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("search.yaml"), "enabled: true\n").unwrap();
        let cfg = config::load(dir.path()).unwrap();

        for (id, expected) in [("code", "code"), ("commits", "commits")] {
            let rt = build_with_config(dir.path(), id, cfg.clone()).unwrap();
            assert_eq!(rt.corpus.id(), expected);
        }
    }

    /// Issues intentionally omitted until Phase 2b-4; must surface as a
    /// clear "unknown corpus" rather than a mysterious nil dispatch.
    #[test]
    fn build_with_config_rejects_issues_until_phase_2b4() {
        let dir = tempfile::tempdir().unwrap();
        // Opt issues in so we pass the gate and hit the factory.
        std::fs::write(
            dir.path().join("search.yaml"),
            "enabled: true\ncorpora:\n  issues:\n    enabled: true\n",
        )
        .unwrap();
        let cfg = config::load(dir.path()).unwrap();

        let err = build_with_config(dir.path(), "issues", cfg).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("unknown corpus"),
            "expected unknown-corpus error, got: {msg}"
        );
    }

    /// Context is enabled-by-override only; when explicitly opted in,
    /// the build path returns a functional ContextCorpus.
    #[test]
    fn build_with_config_context_opt_in() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("search.yaml"),
            "enabled: true\ncorpora:\n  context:\n    enabled: true\n",
        )
        .unwrap();
        let cfg = config::load(dir.path()).unwrap();
        let rt = build_with_config(dir.path(), "context", cfg).unwrap();
        assert_eq!(rt.corpus.id(), "context");
    }
}
