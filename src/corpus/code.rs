//! CodeCorpus: git-tracked source files, chunked for embedding.
//!
//! Ports `search/corpus/code.go`. Like `CommitCorpus`, shells out to
//! `git ls-files` to stay byte-compatible with the Go enumeration.

use super::{chunker, filter, project_hash, truncate_utf8, Corpus, Record};
use anyhow::{anyhow, Context};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug)]
pub struct CodeCorpus {
    pub filter: filter::Filter,
    pub chunk: chunker::ChunkConfig,
}

impl Default for CodeCorpus {
    fn default() -> Self {
        Self {
            filter: filter::Filter::default_values(),
            chunk: chunker::ChunkConfig::default(),
        }
    }
}

const EMBED_HEADER_PREFIX: &str = "File: ";
const MAX_EMBED_CHARS: usize = 12_000;

impl Corpus for CodeCorpus {
    fn id(&self) -> &'static str {
        "code"
    }

    fn collection(&self, project_root: &Path) -> String {
        format!("code-{}", project_hash(project_root))
    }

    fn enumerate(&self, project_root: &Path) -> anyhow::Result<Vec<Record>> {
        let tracked = git_ls_files(project_root)?;
        let mtime = iso_utc_now();

        let mut out = Vec::new();
        for rel in tracked {
            let abs: PathBuf = project_root.join(&rel);
            if !self.filter.accept(&abs) {
                continue;
            }
            let data = match std::fs::read(&abs) {
                Ok(d) => d,
                Err(e) => {
                    tracing::warn!("read {}: {e}", abs.display());
                    continue;
                }
            };
            let hash = hex::encode(Sha256::digest(&data));
            let chunks = chunker::chunk(&data, self.chunk);
            let kind = if chunks.len() > 1 { "chunk" } else { "file" };

            for ch in chunks {
                let mut extra: BTreeMap<String, serde_json::Value> = BTreeMap::new();
                extra.insert("mtime".into(), mtime.clone().into());

                out.push(Record {
                    path: rel.clone(),
                    line_start: ch.line_start,
                    line_end: ch.line_end,
                    kind: kind.into(),
                    content_hash: hash.clone(),
                    extra,
                    embed_text: format_embed_text(&rel, &ch.text),
                });
            }
        }
        Ok(out)
    }
}

/// Prepend a "File: <path>" header so the embedder has path context.
/// Jina v5 benefits from this signal.
fn format_embed_text(path: &str, body: &str) -> String {
    let header = format!("{EMBED_HEADER_PREFIX}{path}\n\n");
    let budget = MAX_EMBED_CHARS.saturating_sub(header.len());
    let trimmed = truncate_utf8(body, budget);
    // Strip trailing NULs that some tools leave at EOF of binary-ish files
    // that still slipped past the filter (e.g. UTF-16 BOM + content).
    let trimmed = trimmed.trim_end_matches('\0');
    format!("{header}{trimmed}")
}

fn git_ls_files(project_root: &Path) -> anyhow::Result<Vec<String>> {
    let out = Command::new("git")
        .arg("-C")
        .arg(project_root)
        .arg("ls-files")
        .output()
        .context("invoking git ls-files")?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(anyhow!("git ls-files: {}", stderr.trim()));
    }
    Ok(String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(|s| s.to_string())
        .collect())
}

fn iso_utc_now() -> String {
    // Minimal RFC3339 formatting without pulling in chrono/time: format as
    // `YYYY-MM-DDTHH:MM:SSZ` using UNIX-epoch math. Good enough for the
    // `mtime` payload field, which exists only so qdrant searches can
    // filter by recency (not for round-tripping to dates).
    let dur = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = dur.as_secs() as i64;
    let (y, mo, d, h, mi, s) = civil_from_unix_secs(secs);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z")
}

/// Compute civil (Y, M, D, H, M, S) from Unix seconds (UTC). Algorithm by
/// Howard Hinnant, public domain.
fn civil_from_unix_secs(secs: i64) -> (i32, u32, u32, u32, u32, u32) {
    let days = secs.div_euclid(86_400);
    let sod = secs.rem_euclid(86_400) as u32;
    let h = sod / 3_600;
    let mi = (sod % 3_600) / 60;
    let s = sod % 60;

    // Hinnant's days-since-epoch → (y, m, d)
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y as i32, m as u32, d as u32, h, mi, s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn git_init(dir: &Path) {
        let run = |args: &[&str]| {
            let out = Command::new("git")
                .args(args)
                .current_dir(dir)
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "git {args:?}: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        };
        run(&["init", "-q"]);
        run(&["config", "user.email", "test@example.com"]);
        run(&["config", "user.name", "test"]);
    }

    #[test]
    fn enumerates_git_tracked_text_files() {
        let dir = tempfile::tempdir().unwrap();
        git_init(dir.path());

        fs::write(
            dir.path().join("hello.go"),
            "package main\n\nfunc main() {}\n",
        )
        .unwrap();
        fs::write(
            dir.path().join("image.png"),
            [0x89, 0x50, 0x4e, 0x47, 0x00, 0x01, 0x02],
        )
        .unwrap();
        let run = |args: &[&str]| {
            let out = Command::new("git")
                .args(args)
                .current_dir(dir.path())
                .output()
                .unwrap();
            assert!(out.status.success(), "git {args:?}");
        };
        run(&["add", "-A"]);
        run(&["commit", "-qm", "init"]);

        let cc = CodeCorpus::default();
        let records = cc.enumerate(dir.path()).unwrap();
        let paths: Vec<_> = records.iter().map(|r| r.path.clone()).collect();
        assert_eq!(paths, vec!["hello.go".to_string()]);
        for r in &records {
            assert!(!r.content_hash.is_empty(), "missing content_hash");
            assert!(!r.embed_text.is_empty(), "missing embed_text");
        }
    }

    #[test]
    fn id_and_collection() {
        let cc = CodeCorpus::default();
        assert_eq!(cc.id(), "code");
        assert!(cc.collection(Path::new(".")).starts_with("code-"));
    }

    #[test]
    fn chunks_large_file() {
        let dir = tempfile::tempdir().unwrap();
        git_init(dir.path());
        fs::write(dir.path().join("big.txt"), "x".repeat(20_000)).unwrap();
        let run = |args: &[&str]| {
            Command::new("git")
                .args(args)
                .current_dir(dir.path())
                .output()
                .unwrap();
        };
        run(&["add", "-A"]);
        run(&["commit", "-qm", "init"]);

        let cc = CodeCorpus {
            filter: filter::Filter {
                max_bytes: 1 << 30,
                excludes: vec![],
            },
            chunk: chunker::ChunkConfig {
                max: 8_000,
                overlap: 200,
            },
        };
        let records = cc.enumerate(dir.path()).unwrap();
        assert!(
            records.len() >= 2,
            "expected ≥2 chunks, got {}",
            records.len()
        );
        for r in &records {
            assert_eq!(r.kind, "chunk");
        }
    }

    #[test]
    fn format_embed_text_prepends_header() {
        let out = format_embed_text("src/foo.rs", "fn foo() {}");
        assert!(out.starts_with("File: src/foo.rs\n\n"));
        assert!(out.contains("fn foo() {}"));
    }

    #[test]
    fn format_embed_text_caps_length() {
        let big = "x".repeat(100_000);
        let out = format_embed_text("foo", &big);
        assert!(out.len() <= MAX_EMBED_CHARS);
    }

    #[test]
    fn civil_from_unix_secs_known_values() {
        // Epoch.
        assert_eq!(civil_from_unix_secs(0), (1970, 1, 1, 0, 0, 0));
        // Exactly one day after epoch.
        assert_eq!(civil_from_unix_secs(86_400), (1970, 1, 2, 0, 0, 0));
        // 2000-02-29 (leap day) at 12:34:56Z = 951_827_696.
        assert_eq!(civil_from_unix_secs(951_827_696), (2000, 2, 29, 12, 34, 56));
        // 2024-12-31T23:59:59Z (end of 2024 leap year) = 1_735_689_599.
        assert_eq!(
            civil_from_unix_secs(1_735_689_599),
            (2024, 12, 31, 23, 59, 59)
        );
    }

    #[test]
    fn iso_utc_now_is_rfc3339_shaped() {
        let now = iso_utc_now();
        // 2026-04-24T05:00:00Z is 20 characters: 4-2-2 T 2:2:2 Z
        assert_eq!(now.len(), 20, "got {now:?}");
        assert!(now.ends_with('Z'), "got {now:?}");
        assert_eq!(&now[4..5], "-");
        assert_eq!(&now[7..8], "-");
        assert_eq!(&now[10..11], "T");
        assert_eq!(&now[13..14], ":");
        assert_eq!(&now[16..17], ":");
    }
}
