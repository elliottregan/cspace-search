//! CommitCorpus: git commit history (subject + body + diff summary).
//!
//! Ports `search/corpus/commits.go`. Shells out to `git` rather than
//! linking libgit2 — matches the Go implementation byte-for-byte, keeps
//! the dependency surface zero, and the parsing cost is negligible
//! against a 500-commit default limit.

use crate::corpus::{project_hash, truncate_utf8, Corpus, Record};
use anyhow::{anyhow, Context};
use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;

/// Embed-text cap. Jina v5 retrieval supports 8K tokens; 12000 chars
/// ≈ 3000 tokens with headroom for the task prefix.
const MAX_EMBED_CHARS: usize = 12_000;

/// Default commit limit if `CommitCorpus::limit` is 0.
const DEFAULT_LIMIT: usize = 500;

/// Character cap on the diff snippet appended to embed text.
const MAX_DIFF_SNIPPET: usize = 2_000;

#[derive(Debug, Default)]
pub struct CommitCorpus {
    pub limit: usize,
}

impl Corpus for CommitCorpus {
    fn id(&self) -> &'static str {
        "commits"
    }

    fn collection(&self, project_root: &Path) -> String {
        format!("commits-{}", project_hash(project_root))
    }

    fn enumerate(&self, project_root: &Path) -> anyhow::Result<Vec<Record>> {
        let limit = if self.limit == 0 {
            DEFAULT_LIMIT
        } else {
            self.limit
        };
        let commits = list_commits(project_root, limit)?;

        let mut out = Vec::with_capacity(commits.len());
        for cm in commits {
            let mut extra: BTreeMap<String, serde_json::Value> = BTreeMap::new();
            extra.insert("hash".into(), cm.hash.clone().into());
            extra.insert("date".into(), cm.date.clone().into());
            extra.insert("subject".into(), cm.subject.clone().into());

            let embed_text = commit_embed_text(&cm);
            out.push(Record {
                path: cm.hash,
                line_start: 0,
                line_end: 0,
                kind: "commit".into(),
                content_hash: String::new(),
                extra,
                embed_text,
            });
        }
        Ok(out)
    }
}

struct CommitRaw {
    hash: String,
    /// YYYY-MM-DD portion of the author date.
    date: String,
    subject: String,
    body: String,
    diff_summary: String,
}

fn commit_embed_text(c: &CommitRaw) -> String {
    let mut parts: Vec<&str> = Vec::with_capacity(4);
    parts.push(&c.subject);
    if !c.body.is_empty() {
        parts.push(&c.body);
    }
    if !c.diff_summary.is_empty() {
        parts.push("---");
        parts.push(&c.diff_summary);
    }
    let text = parts.join("\n\n");
    truncate_utf8(&text, MAX_EMBED_CHARS).to_string()
}

fn git(repo_path: &Path, args: &[&str]) -> anyhow::Result<String> {
    let out = Command::new("git")
        .args(args)
        .current_dir(repo_path)
        .output()
        .context("invoking git")?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(anyhow!("git {}: {}", args.join(" "), stderr.trim()));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

fn list_commits(repo_path: &Path, limit: usize) -> anyhow::Result<Vec<CommitRaw>> {
    // Rare delimiters so bodies can safely contain pipes, colons, or any
    // single printable character. Matches the Go version byte-for-byte.
    const SEP: &str = "|||COMMIT|||";
    const END: &str = "|||END|||";
    let format = format!("--format={SEP}%H|%aI|%s|%b{END}");
    let limit_flag = format!("-n{limit}");

    let log_out = git(repo_path, &["log", &limit_flag, &format])?;

    let mut records = Vec::new();
    for block in log_out.split(END) {
        let block = block.trim();
        if block.is_empty() {
            continue;
        }
        let block = block.strip_prefix(SEP).unwrap_or(block);
        // splitn(4, '|') — body may contain pipes.
        let parts: Vec<&str> = block.splitn(4, '|').collect();
        if parts.len() < 3 {
            continue;
        }
        let hash = parts[0].trim().to_string();
        let date_str = parts[1].trim();
        let subject = parts[2].trim().to_string();
        let body = parts
            .get(3)
            .map(|s| s.trim().to_string())
            .unwrap_or_default();

        let date = extract_date(date_str);
        let diff_summary = commit_diff_summary(repo_path, &hash).unwrap_or_default();

        records.push(CommitRaw {
            hash,
            date,
            subject,
            body,
            diff_summary,
        });
    }
    Ok(records)
}

/// Extract the date prefix from an RFC3339 timestamp. We only want
/// `YYYY-MM-DD` (matches the Go version's `Format("2006-01-02")`).
fn extract_date(iso: &str) -> String {
    iso.split('T')
        .next()
        .filter(|d| d.len() == 10)
        .unwrap_or("")
        .to_string()
}

fn commit_diff_summary(repo_path: &Path, hash: &str) -> anyhow::Result<String> {
    let stat = git(repo_path, &["show", "--stat", "--no-patch", hash])?;
    // Keep the file-stat rows (contain `|`) + the summary line (contains
    // "changed"). Drop the header.
    let stat_body: Vec<&str> = stat
        .trim()
        .lines()
        .filter(|l| l.contains('|') || l.contains("changed"))
        .collect();
    let stat_joined = stat_body.join("\n");

    // First commit has no parent; fall back to stat-only.
    match git(repo_path, &["diff", &format!("{hash}^"), hash, "--", "."]) {
        Ok(diff) => {
            let snippet = truncate_utf8(&diff, MAX_DIFF_SNIPPET);
            Ok(format!("{stat_joined}\n{snippet}"))
        }
        Err(_) => Ok(stat_joined),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Shape-only contract: Enumerate emits well-formed Records with
    /// kind=commit. Asserting specific commit subjects kept breaking as
    /// work landed on top, so we match the Go suite's "sturdier contract".
    #[test]
    fn enumerate_emits_commit_records() {
        let cc = CommitCorpus { limit: 10 };
        // Run against the cspace-search repo itself.
        let records = cc.enumerate(Path::new(".")).expect("enumerate");
        assert!(
            !records.is_empty(),
            "expected at least one commit record from Enumerate"
        );
        for rec in &records {
            assert_eq!(rec.kind, "commit");
            assert!(!rec.path.is_empty(), "record missing hash: {rec:?}");
            assert!(!rec.embed_text.is_empty(), "record missing embed text");
        }
    }

    #[test]
    fn collection_name_prefix() {
        let cc = CommitCorpus::default();
        let got = cc.collection(Path::new("."));
        assert!(got.starts_with("commits-"), "got {got:?}");
    }

    #[test]
    fn extract_date_happy_path() {
        assert_eq!(extract_date("2026-04-23T22:20:00-06:00"), "2026-04-23");
        assert_eq!(extract_date(""), "");
        assert_eq!(extract_date("bogus"), "");
    }

    #[test]
    fn embed_text_includes_subject_and_body() {
        let cm = CommitRaw {
            hash: "abc".into(),
            date: "2026-04-23".into(),
            subject: "fix thing".into(),
            body: "because reason".into(),
            diff_summary: String::new(),
        };
        let t = commit_embed_text(&cm);
        assert!(t.contains("fix thing"));
        assert!(t.contains("because reason"));
        assert!(!t.contains("---"), "no diff separator when diff empty");
    }

    #[test]
    fn embed_text_appends_diff_with_separator() {
        let cm = CommitRaw {
            hash: "abc".into(),
            date: "2026-04-23".into(),
            subject: "s".into(),
            body: String::new(),
            diff_summary: "+ added line".into(),
        };
        let t = commit_embed_text(&cm);
        assert!(t.contains("---"));
        assert!(t.contains("+ added line"));
    }
}
