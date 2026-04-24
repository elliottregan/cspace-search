//! FileCorpus: walk a set of paths, filter, (optionally) chunk, and
//! emit Records with a templated header.
//!
//! Replaces the two earlier corpora (`CodeCorpus` + `ContextCorpus`)
//! with a single config-driven implementation. Users can now declare
//! new corpora in `search.yaml` without writing Rust:
//!
//! ```yaml
//! corpora:
//!   docs:
//!     enabled: true
//!     type: files
//!     source: walk                # .gitignore-aware
//!     chunk: { max: 12000, overlap: 200 }
//!     record_kind: doc
//!     embed_header: "Documentation: {path}\n\n"
//!     path_groups:
//!       - include: ["docs/**/*.md", "README.md"]
//! ```
//!
//! Sources:
//! - `git-ls-files` — delegates to `git ls-files` for the seed list
//!   (matches the old CodeCorpus). Requires `git` on PATH.
//! - `filesystem` — plain `read_dir` of the project (matches the old
//!   ContextCorpus). Cheap, but walks vendored directories you might
//!   not want.
//! - `walk` — `ignore` crate, .gitignore-aware. New option for
//!   corpora that span tracked + untracked files but still want
//!   vendored-directory exclusion for free.

use super::chunker::{chunk, ChunkConfig, ChunkOut};
use super::filter::{glob_match, Filter};
use super::{project_hash, truncate_utf8, Corpus, Record};
use crate::config::{ChunkSpec, CorpusConfig, PathGroupSpec};
use anyhow::{anyhow, Context, Result};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Default embed-text truncation budget. Jina v5 retrieval accepts
/// ~3K tokens; 12000 chars leaves headroom for the task prefix.
pub const DEFAULT_MAX_EMBED_CHARS: usize = 12_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileSource {
    /// `git ls-files` — every tracked file, once.
    GitLsFiles,
    /// `std::fs::read_dir` walk, respecting only explicit excludes.
    Filesystem,
    /// `ignore` crate walk — respects `.gitignore`, `.ignore`, and
    /// global git excludes, plus our own excludes on top.
    Walk,
}

#[derive(Debug, Clone)]
pub struct PathGroup {
    pub include: Vec<String>,
    /// Per-group `kind` override. Falls back to the corpus default.
    pub kind: Option<String>,
    /// Per-group chunking override. Falls back to the corpus default.
    pub chunk: Option<ChunkConfig>,
    /// Per-group extra fields, added to `Record.extra` after template
    /// expansion.
    pub extra: BTreeMap<String, String>,
}

#[derive(Debug)]
pub struct FileCorpus {
    pub id: String,
    pub source: FileSource,
    pub filter: Filter,
    pub chunk: Option<ChunkConfig>,
    pub path_groups: Vec<PathGroup>,
    pub embed_header: String,
    pub default_record_kind: String,
    pub max_embed_chars: usize,
}

impl FileCorpus {
    /// Build a corpus from its `id` and resolved `CorpusConfig`. This
    /// is the sole public constructor outside tests — the factory in
    /// `config::runtime` calls it.
    pub fn from_config(id: &str, cfg: &CorpusConfig) -> Result<Self> {
        let source = match cfg.source.as_deref().unwrap_or("git-ls-files") {
            "git-ls-files" => FileSource::GitLsFiles,
            "filesystem" => FileSource::Filesystem,
            "walk" => FileSource::Walk,
            other => {
                return Err(anyhow!(
                    "unknown source {other:?} for corpus {id:?}; expected one of git-ls-files | filesystem | walk"
                ))
            }
        };
        let filter = Filter {
            max_bytes: cfg.max_bytes.max(0) as u64,
            excludes: cfg.excludes.clone(),
        };
        let corpus_chunk = cfg.chunk.as_ref().map(chunk_from_spec);

        // When no path_groups are declared, synthesise one that
        // accepts everything the source enumerates. This keeps the
        // "simple case" YAML minimal: `code` needs no path_groups.
        let path_groups: Vec<PathGroup> = if cfg.path_groups.is_empty() {
            vec![PathGroup {
                include: vec![],
                kind: None,
                chunk: None,
                extra: BTreeMap::new(),
            }]
        } else {
            cfg.path_groups.iter().map(path_group_from_spec).collect()
        };

        Ok(Self {
            id: id.to_string(),
            source,
            filter,
            chunk: corpus_chunk,
            path_groups,
            embed_header: cfg
                .embed_header
                .clone()
                .unwrap_or_else(|| "{kind}: {path}\n\n".to_string()),
            default_record_kind: cfg.record_kind.clone().unwrap_or_else(|| "file".into()),
            max_embed_chars: cfg.max_embed_chars.unwrap_or(DEFAULT_MAX_EMBED_CHARS),
        })
    }
}

fn chunk_from_spec(s: &ChunkSpec) -> ChunkConfig {
    ChunkConfig {
        max: s.max,
        overlap: s.overlap,
    }
}

fn path_group_from_spec(s: &PathGroupSpec) -> PathGroup {
    PathGroup {
        include: s.include.clone(),
        kind: s.kind.clone(),
        chunk: s.chunk.as_ref().map(chunk_from_spec),
        extra: s.extra.clone(),
    }
}

impl Corpus for FileCorpus {
    fn id(&self) -> &'static str {
        // Corpus::id() returns &'static str for the Go-parity trait;
        // we leak the owned String once per process so callers keep
        // the old signature. The corpus is long-lived (one per
        // run_loop), so the leak is bounded and small.
        Box::leak(self.id.clone().into_boxed_str())
    }

    fn collection(&self, project_root: &Path) -> String {
        format!("{}-{}", self.id, project_hash(project_root))
    }

    fn kinds(&self) -> Vec<String> {
        // Union over path_groups' kind overrides, falling back to
        // the corpus default. Sorted + deduped so the MCP layer can
        // compare lists byte-for-byte across calls.
        let mut set: std::collections::BTreeSet<String> =
            std::collections::BTreeSet::new();
        let mut any_default = false;
        for g in &self.path_groups {
            match &g.kind {
                Some(k) => {
                    set.insert(k.clone());
                }
                None => any_default = true,
            }
        }
        if any_default {
            set.insert(self.default_record_kind.clone());
        }
        set.into_iter().collect()
    }

    fn enumerate(&self, project_root: &Path) -> Result<Vec<Record>> {
        let candidates = enumerate_candidates(self.source, project_root, &self.filter)?;

        let mut out = Vec::new();
        for rel in candidates {
            let Some(group) = self.match_group(&rel) else {
                continue;
            };
            let abs: PathBuf = project_root.join(&rel);
            // Run the null-byte + size + glob-exclude filter. The
            // walker already excluded .gitignore-hidden paths; this
            // catches size and binary content.
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
            let chunk_cfg = group
                .chunk
                .or(self.chunk)
                .unwrap_or(ChunkConfig { max: 0, overlap: 0 });
            let chunks: Vec<ChunkOut> = if chunk_cfg.max == 0 {
                // No chunking requested: a single whole-file chunk
                // with the full line span.
                let body = String::from_utf8_lossy(&data).into_owned();
                let end = body.matches('\n').count() as u32 + 1;
                vec![ChunkOut {
                    text: body,
                    line_start: 1,
                    line_end: end.max(1),
                }]
            } else {
                chunk(&data, chunk_cfg)
            };
            let resolved_kind = group
                .kind
                .clone()
                .unwrap_or_else(|| self.default_record_kind.clone());
            let multi = chunks.len() > 1;
            for ch in chunks {
                let kind = if multi { "chunk" } else { &resolved_kind };

                // Template variables.
                let mut vars: BTreeMap<String, String> = BTreeMap::new();
                vars.insert("path".into(), rel.clone());
                vars.insert("kind".into(), kind.to_string());
                vars.insert("basename".into(), basename(&rel));
                vars.insert("basename_no_ext".into(), basename_no_ext(&rel));
                // Per-group extras, each run through the same
                // template engine so `subkind: "{basename_no_ext}"`
                // works.
                let mut extra_values: BTreeMap<String, serde_json::Value> = BTreeMap::new();
                for (k, tmpl) in &group.extra {
                    let rendered = render_template(tmpl, &vars);
                    // Let users reference group-extra values from
                    // `embed_header`: add each one to vars as it
                    // resolves.
                    vars.insert(k.clone(), rendered.clone());
                    extra_values.insert(k.clone(), rendered.into());
                }

                let header = render_template(&self.embed_header, &vars);
                let body_budget = self.max_embed_chars.saturating_sub(header.len());
                let body = truncate_utf8(&ch.text, body_budget);
                // Strip trailing NULs that might slip past the filter
                // on multi-byte-encoded files — matches the old
                // CodeCorpus behaviour.
                let body = body.trim_end_matches('\0');
                let embed_text = format!("{header}{body}");

                // Content hash is over the *final* embed_text, not the
                // source bytes. This lets the indexer's skip cache
                // detect changes to the header template, record_kind,
                // or any path-group `extra` value — all of which alter
                // the embedding input without changing the underlying
                // file. A file-level change still invalidates every
                // affected chunk since each chunk's body slice differs.
                let content_hash = hex::encode(Sha256::digest(embed_text.as_bytes()));

                out.push(Record {
                    path: rel.clone(),
                    line_start: ch.line_start,
                    line_end: ch.line_end,
                    kind: kind.to_string(),
                    content_hash,
                    extra: extra_values,
                    embed_text,
                });
            }
        }
        Ok(out)
    }
}

impl FileCorpus {
    /// First PathGroup whose includes match `rel`. A group with an
    /// empty `include` list matches everything — this is the default
    /// when a user declares no groups (e.g. the `code` corpus).
    fn match_group(&self, rel: &str) -> Option<&PathGroup> {
        self.path_groups
            .iter()
            .find(|g| g.include.is_empty() || g.include.iter().any(|pat| glob_match(pat, rel)))
    }
}

fn enumerate_candidates(
    source: FileSource,
    project_root: &Path,
    filter: &Filter,
) -> Result<Vec<String>> {
    match source {
        FileSource::GitLsFiles => git_ls_files(project_root),
        FileSource::Filesystem => filesystem_walk(project_root),
        FileSource::Walk => ignore_walk(project_root, filter),
    }
}

fn git_ls_files(project_root: &Path) -> Result<Vec<String>> {
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

fn filesystem_walk(project_root: &Path) -> Result<Vec<String>> {
    // Plain recursive walk — does NOT respect .gitignore. Use `walk`
    // when you want that. This matches the old ContextCorpus
    // behaviour, which iterated a fixed set of subdirs.
    let mut out = Vec::new();
    let mut stack: Vec<PathBuf> = vec![project_root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = match std::fs::read_dir(&dir) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let file_type = match entry.file_type() {
                Ok(t) => t,
                Err(_) => continue,
            };
            if file_type.is_dir() {
                stack.push(path);
            } else if file_type.is_file() {
                if let Ok(rel) = path.strip_prefix(project_root) {
                    out.push(rel.to_string_lossy().into_owned());
                }
            }
        }
    }
    Ok(out)
}

fn ignore_walk(project_root: &Path, _filter: &Filter) -> Result<Vec<String>> {
    let mut out = Vec::new();
    for entry in ignore::WalkBuilder::new(project_root).build() {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        if !entry.file_type().is_some_and(|t| t.is_file()) {
            continue;
        }
        if let Ok(rel) = entry.path().strip_prefix(project_root) {
            out.push(rel.to_string_lossy().into_owned());
        }
    }
    Ok(out)
}

/// Substitute `{key}` placeholders in `tmpl` using `vars`. Unknown
/// keys are left as-is (visible to the user so typos are obvious).
fn render_template(tmpl: &str, vars: &BTreeMap<String, String>) -> String {
    let mut out = tmpl.to_string();
    for (k, v) in vars {
        out = out.replace(&format!("{{{k}}}"), v);
    }
    out
}

fn basename(rel: &str) -> String {
    Path::new(rel)
        .file_name()
        .and_then(|os| os.to_str())
        .unwrap_or(rel)
        .to_string()
}

fn basename_no_ext(rel: &str) -> String {
    let b = basename(rel);
    match b.rsplit_once('.') {
        Some((stem, _ext)) => stem.to_string(),
        None => b,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::process::Command as Cmd;

    fn git_init(dir: &Path) {
        let run = |args: &[&str]| {
            let out = Cmd::new("git")
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

    fn commit_all(dir: &Path) {
        let run = |args: &[&str]| {
            Cmd::new("git")
                .args(args)
                .current_dir(dir)
                .output()
                .unwrap();
        };
        run(&["add", "-A"]);
        run(&["commit", "-qm", "seed"]);
    }

    fn code_config() -> CorpusConfig {
        CorpusConfig {
            enabled: true,
            max_bytes: 204_800,
            excludes: vec!["vendor/**".into(), "*.lock".into(), "*.png".into()],
            chunk: Some(ChunkSpec {
                max: 12_000,
                overlap: 200,
            }),
            embed_header: Some("File: {path}\n\n".into()),
            record_kind: Some("file".into()),
            source: Some("git-ls-files".into()),
            type_name: Some("files".into()),
            ..CorpusConfig::default()
        }
    }

    fn context_config() -> CorpusConfig {
        CorpusConfig {
            enabled: true,
            source: Some("filesystem".into()),
            type_name: Some("files".into()),
            record_kind: Some("context".into()),
            embed_header: Some("Context ({subkind}): {path}\n\n".into()),
            path_groups: vec![
                PathGroupSpec {
                    include: vec![
                        ".cspace/context/direction.md".into(),
                        ".cspace/context/principles.md".into(),
                        ".cspace/context/roadmap.md".into(),
                    ],
                    kind: Some("context".into()),
                    chunk: None,
                    extra: [("subkind".to_string(), "{basename_no_ext}".to_string())]
                        .into_iter()
                        .collect(),
                },
                PathGroupSpec {
                    include: vec![".cspace/context/findings/**/*.md".into()],
                    kind: Some("finding".into()),
                    chunk: None,
                    extra: BTreeMap::new(),
                },
            ],
            ..CorpusConfig::default()
        }
    }

    #[test]
    fn code_corpus_enumerates_git_tracked_text_files() {
        let dir = tempfile::tempdir().unwrap();
        git_init(dir.path());
        fs::write(dir.path().join("hello.rs"), "fn main() {}\n").unwrap();
        fs::write(
            dir.path().join("image.png"),
            [0x89, 0x50, 0x4e, 0x47, 0x00, 0x01, 0x02],
        )
        .unwrap();
        commit_all(dir.path());

        let corpus = FileCorpus::from_config("code", &code_config()).unwrap();
        let records = corpus.enumerate(dir.path()).unwrap();
        let paths: Vec<_> = records.iter().map(|r| r.path.clone()).collect();
        assert_eq!(paths, vec!["hello.rs".to_string()]);
        let r = &records[0];
        assert!(!r.content_hash.is_empty());
        assert!(r.embed_text.starts_with("File: hello.rs\n\n"));
        assert!(r.embed_text.contains("fn main() {}"));
        assert_eq!(r.kind, "file");
    }

    #[test]
    fn code_corpus_chunks_large_files() {
        let dir = tempfile::tempdir().unwrap();
        git_init(dir.path());
        fs::write(dir.path().join("big.txt"), "x".repeat(20_000)).unwrap();
        commit_all(dir.path());

        let mut cfg = code_config();
        cfg.max_bytes = 1 << 30;
        cfg.excludes.clear();
        cfg.chunk = Some(ChunkSpec {
            max: 8_000,
            overlap: 200,
        });
        let corpus = FileCorpus::from_config("code", &cfg).unwrap();
        let records = corpus.enumerate(dir.path()).unwrap();
        assert!(records.len() >= 2);
        for r in &records {
            assert_eq!(r.kind, "chunk");
        }
    }

    #[test]
    fn context_corpus_emits_kinds_per_path_group() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = dir.path().join(".cspace").join("context");
        fs::create_dir_all(ctx.join("findings")).unwrap();
        fs::write(ctx.join("principles.md"), "# Principles\n").unwrap();
        fs::write(ctx.join("roadmap.md"), "# Roadmap\n").unwrap();
        fs::write(
            ctx.join("findings/2026-04-13-thing.md"),
            "# Finding\nfound it.\n",
        )
        .unwrap();

        let corpus = FileCorpus::from_config("context", &context_config()).unwrap();
        let records = corpus.enumerate(dir.path()).unwrap();
        let by_path: BTreeMap<String, Record> =
            records.into_iter().map(|r| (r.path.clone(), r)).collect();

        let principles = &by_path[".cspace/context/principles.md"];
        assert_eq!(principles.kind, "context");
        assert_eq!(
            principles.extra.get("subkind").and_then(|v| v.as_str()),
            Some("principles")
        );
        assert!(principles
            .embed_text
            .starts_with("Context (principles): .cspace/context/principles.md\n\n"));

        let finding = &by_path[".cspace/context/findings/2026-04-13-thing.md"];
        assert_eq!(finding.kind, "finding");
        // No subkind for subdirectory groups.
        assert!(!finding.extra.contains_key("subkind"));
    }

    #[test]
    fn walk_source_respects_gitignore() {
        let dir = tempfile::tempdir().unwrap();
        git_init(dir.path());
        fs::write(dir.path().join(".gitignore"), "ignored.txt\n").unwrap();
        fs::write(dir.path().join("keep.txt"), "visible").unwrap();
        fs::write(dir.path().join("ignored.txt"), "hidden").unwrap();

        let mut cfg = CorpusConfig {
            enabled: true,
            source: Some("walk".into()),
            max_bytes: 1 << 30,
            embed_header: Some("Walk: {path}\n\n".into()),
            record_kind: Some("walked".into()),
            ..CorpusConfig::default()
        };
        // One group, include all.
        cfg.path_groups = vec![PathGroupSpec {
            include: vec!["**/*".into()],
            ..PathGroupSpec::default()
        }];
        let corpus = FileCorpus::from_config("w", &cfg).unwrap();
        let paths: Vec<String> = corpus
            .enumerate(dir.path())
            .unwrap()
            .into_iter()
            .map(|r| r.path)
            .collect();
        assert!(paths.contains(&"keep.txt".into()));
        assert!(!paths.contains(&"ignored.txt".into()));
    }

    #[test]
    fn embed_header_interpolates_extra_keys() {
        let mut vars = BTreeMap::new();
        vars.insert("path".into(), "a.rs".into());
        vars.insert("subkind".into(), "direction".into());
        assert_eq!(
            render_template("Context ({subkind}): {path}\n\n", &vars),
            "Context (direction): a.rs\n\n"
        );
    }

    /// Header-template changes must invalidate the content_hash skip
    /// cache. Pre-refactor the hash was sha256(file_bytes), so a
    /// header change silently kept stale vectors in the store. Post
    /// refactor the hash is sha256(embed_text) — the exact string
    /// fed to the embedder — so any template change flips the hash.
    #[test]
    fn content_hash_covers_header_template() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = dir.path().join(".cspace/context");
        fs::create_dir_all(&ctx).unwrap();
        fs::write(ctx.join("principles.md"), "keep it simple").unwrap();

        let mut cfg = context_config();
        cfg.embed_header = Some("First: {path}\n\n".into());
        let hash1 = FileCorpus::from_config("context", &cfg)
            .unwrap()
            .enumerate(dir.path())
            .unwrap()[0]
            .content_hash
            .clone();

        cfg.embed_header = Some("Second: {path}\n\n".into());
        let hash2 = FileCorpus::from_config("context", &cfg)
            .unwrap()
            .enumerate(dir.path())
            .unwrap()[0]
            .content_hash
            .clone();

        assert_ne!(hash1, hash2, "header change must invalidate content_hash");
    }

    /// Conversely: with the same config, two chunks of the same file
    /// must produce stable hashes across runs. This is the property
    /// the indexer's skip cache actually relies on.
    #[test]
    fn content_hash_stable_across_runs_with_same_config() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = dir.path().join(".cspace/context");
        fs::create_dir_all(&ctx).unwrap();
        fs::write(ctx.join("principles.md"), "keep it simple").unwrap();

        let cfg = context_config();
        let first: Vec<String> = FileCorpus::from_config("context", &cfg)
            .unwrap()
            .enumerate(dir.path())
            .unwrap()
            .into_iter()
            .map(|r| r.content_hash)
            .collect();
        let second: Vec<String> = FileCorpus::from_config("context", &cfg)
            .unwrap()
            .enumerate(dir.path())
            .unwrap()
            .into_iter()
            .map(|r| r.content_hash)
            .collect();
        assert_eq!(first, second);
    }

    #[test]
    fn basename_helpers() {
        assert_eq!(basename("a/b/c.md"), "c.md");
        assert_eq!(basename_no_ext("a/b/c.md"), "c");
        assert_eq!(basename_no_ext("README"), "README");
    }

    #[test]
    fn unknown_source_rejects_cleanly() {
        let cfg = CorpusConfig {
            source: Some("bogus".into()),
            ..CorpusConfig::default()
        };
        let err = FileCorpus::from_config("x", &cfg).unwrap_err();
        assert!(err.to_string().contains("unknown source"));
    }

    #[test]
    fn default_record_kind_when_omitted() {
        let cfg = CorpusConfig::default();
        let corpus = FileCorpus::from_config("x", &cfg).unwrap();
        assert_eq!(corpus.default_record_kind, "file");
    }

    #[test]
    fn id_and_collection() {
        let cfg = code_config();
        let corpus = FileCorpus::from_config("code", &cfg).unwrap();
        assert_eq!(corpus.id(), "code");
        assert!(corpus.collection(Path::new(".")).starts_with("code-"));
    }
}
