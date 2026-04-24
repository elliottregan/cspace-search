//! Single-chokepoint gating for `search`'s "is this corpus runnable?"
//! question. Every CLI query, init loop, and MCP tool routes through
//! [`resolve_corpus`] so a disabled search or disabled corpus produces
//! one consistent, actionable error.
//!
//! Ported from `search/config/runtime.go`. The Go version returned a
//! `Runtime` bundle that paired the config with an instantiated
//! `corpus.Corpus` trait object; the trait lives in a later phase, so
//! this module stops at the gating decision. Callers that need a corpus
//! instance build it themselves until the corpus module lands.

use super::{Config, CorpusConfig};
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SearchError {
    #[error("search not configured for this project (set `enabled: true` in {0}/search.yaml to activate)")]
    SearchDisabled(String),

    #[error("{corpus:?}: corpus disabled in search.yaml (set corpora.{corpus}.enabled=true in search.yaml to enable)")]
    CorpusDisabled { corpus: String },

    #[error("unknown corpus {corpus:?} (known: code, commits, context, issues)")]
    UnknownCorpus { corpus: String },
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
}
