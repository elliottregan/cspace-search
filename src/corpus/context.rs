//! ContextCorpus: layered planning artifacts under `.cspace/context/`.
//!
//! Indexes `direction.md`, `principles.md`, `roadmap.md`, and the
//! `findings/`, `decisions/`, `discoveries/` subdirectories. No
//! chunking — individual context files are small; truncate to the
//! embed limit and move on. Ports `search/corpus/context.go`.

use super::{project_hash, truncate_utf8, Corpus, Record};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Debug)]
pub struct ContextCorpus;

const MAX_EMBED_CHARS: usize = 12_000;

const TOP_LEVEL_FILES: &[&str] = &["direction.md", "principles.md", "roadmap.md"];

/// Subdir → per-record `kind`.
const SUBDIRS: &[(&str, &str)] = &[
    ("findings", "finding"),
    ("decisions", "decision"),
    ("discoveries", "discovery"),
];

impl Corpus for ContextCorpus {
    fn id(&self) -> &'static str {
        "context"
    }

    fn collection(&self, project_root: &Path) -> String {
        format!("context-{}", project_hash(project_root))
    }

    fn enumerate(&self, project_root: &Path) -> anyhow::Result<Vec<Record>> {
        let ctx_dir = project_root.join(".cspace").join("context");
        let mut out = Vec::new();

        // Top-level files: kind="context", subkind = basename without .md.
        for name in TOP_LEVEL_FILES {
            let abs = ctx_dir.join(name);
            let data = match std::fs::read(&abs) {
                Ok(d) => d,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
                Err(e) => {
                    tracing::warn!("read {}: {e}", abs.display());
                    continue;
                }
            };
            let subkind = name.trim_end_matches(".md");
            let rel = format!(".cspace/context/{name}");
            out.push(build_record(&rel, subkind, "context", &data));
        }

        // Subdirectories: each has its own kind.
        for (subdir, kind) in SUBDIRS {
            let sub_path = ctx_dir.join(subdir);
            let entries = match std::fs::read_dir(&sub_path) {
                Ok(e) => e,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
                Err(e) => {
                    tracing::warn!("readdir {}: {e}", sub_path.display());
                    continue;
                }
            };
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if !name_str.ends_with(".md") {
                    continue;
                }
                let file_type = match entry.file_type() {
                    Ok(t) => t,
                    Err(_) => continue,
                };
                if file_type.is_dir() {
                    continue;
                }
                let abs = entry.path();
                let data = match std::fs::read(&abs) {
                    Ok(d) => d,
                    Err(e) => {
                        tracing::warn!("read {}: {e}", abs.display());
                        continue;
                    }
                };
                let rel = format!(".cspace/context/{subdir}/{name_str}");
                out.push(build_record(&rel, kind, kind, &data));
            }
        }

        Ok(out)
    }
}

fn build_record(rel_path: &str, label_for_header: &str, kind: &str, data: &[u8]) -> Record {
    let hash = hex::encode(Sha256::digest(data));
    let body = String::from_utf8_lossy(data);
    let mut extra: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    if kind == "context" {
        // Top-level files carry a subkind for disambiguation.
        extra.insert("subkind".into(), label_for_header.into());
    }
    Record {
        path: rel_path.into(),
        line_start: 0,
        line_end: 0,
        kind: kind.into(),
        content_hash: hash,
        extra,
        embed_text: format_embed_text(label_for_header, rel_path, &body),
    }
}

fn format_embed_text(label: &str, path: &str, body: &str) -> String {
    let header = format!("Context ({label}): {path}\n\n");
    let budget = MAX_EMBED_CHARS.saturating_sub(header.len());
    let trimmed = truncate_utf8(body, budget);
    format!("{header}{trimmed}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::fs;

    fn make_context_dir(dir: &Path, files: &[(&str, &str)]) {
        let ctx_dir = dir.join(".cspace").join("context");
        fs::create_dir_all(&ctx_dir).unwrap();
        for (rel, content) in files {
            let abs = ctx_dir.join(rel);
            fs::create_dir_all(abs.parent().unwrap()).unwrap();
            fs::write(&abs, content).unwrap();
        }
    }

    fn by_path(records: Vec<Record>) -> HashMap<String, Record> {
        records.into_iter().map(|r| (r.path.clone(), r)).collect()
    }

    #[test]
    fn enumerates_all_artifact_types() {
        let dir = tempfile::tempdir().unwrap();
        make_context_dir(
            dir.path(),
            &[
                ("principles.md", "# Principles\nKeep it simple."),
                (
                    "findings/2026-04-13-some-finding.md",
                    "# Finding\nSomething was found.",
                ),
                (
                    "decisions/2026-04-10-use-go.md",
                    "# Decision\nUse Go for the CLI.",
                ),
                (
                    "discoveries/2026-04-12-perf-insight.md",
                    "# Discovery\nPerf insight here.",
                ),
            ],
        );

        let records = ContextCorpus.enumerate(dir.path()).unwrap();
        assert_eq!(records.len(), 4);
        let by = by_path(records);

        let r = by
            .get(".cspace/context/principles.md")
            .expect("missing principles");
        assert_eq!(r.kind, "context");
        assert_eq!(r.extra.get("subkind").unwrap(), "principles");

        assert_eq!(
            by[".cspace/context/findings/2026-04-13-some-finding.md"].kind,
            "finding"
        );
        assert_eq!(
            by[".cspace/context/decisions/2026-04-10-use-go.md"].kind,
            "decision"
        );
        assert_eq!(
            by[".cspace/context/discoveries/2026-04-12-perf-insight.md"].kind,
            "discovery"
        );

        for r in by.values() {
            assert!(!r.content_hash.is_empty());
            assert!(!r.embed_text.is_empty());
        }
    }

    #[test]
    fn missing_subdirectories_are_skipped() {
        let dir = tempfile::tempdir().unwrap();
        make_context_dir(dir.path(), &[("principles.md", "# P")]);
        let records = ContextCorpus.enumerate(dir.path()).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].kind, "context");
        assert_eq!(records[0].path, ".cspace/context/principles.md");
    }

    #[test]
    fn empty_context_dir() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir_all(dir.path().join(".cspace").join("context")).unwrap();
        let records = ContextCorpus.enumerate(dir.path()).unwrap();
        assert!(records.is_empty());
    }

    #[test]
    fn no_context_dir_at_all() {
        let dir = tempfile::tempdir().unwrap();
        let records = ContextCorpus.enumerate(dir.path()).unwrap();
        assert!(records.is_empty());
    }

    #[test]
    fn id_and_collection() {
        assert_eq!(ContextCorpus.id(), "context");
        assert!(ContextCorpus
            .collection(Path::new("."))
            .starts_with("context-"));
    }

    #[test]
    fn embed_text_format() {
        let dir = tempfile::tempdir().unwrap();
        make_context_dir(dir.path(), &[("direction.md", "We are heading north.")]);
        let records = ContextCorpus.enumerate(dir.path()).unwrap();
        assert_eq!(records.len(), 1);
        let t = &records[0].embed_text;
        assert!(t.starts_with("Context (direction): .cspace/context/direction.md\n\n"));
        assert!(t.contains("We are heading north."));
    }

    #[test]
    fn roadmap_and_direction_get_subkind() {
        let dir = tempfile::tempdir().unwrap();
        make_context_dir(
            dir.path(),
            &[("direction.md", "# Direction"), ("roadmap.md", "# Roadmap")],
        );
        let records = ContextCorpus.enumerate(dir.path()).unwrap();
        let by = by_path(records);
        assert_eq!(
            by[".cspace/context/direction.md"].extra["subkind"],
            "direction"
        );
        assert_eq!(by[".cspace/context/roadmap.md"].extra["subkind"], "roadmap");
    }
}
