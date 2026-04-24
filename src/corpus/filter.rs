//! Filter: decides whether a file should be indexed.
//!
//! Ports `search/corpus/code_filter.go`. Preserves the Go version's
//! custom `**` handling so multi-segment-prefix excludes (e.g.
//! `docs/superpowers/specs/**`) match both relative and absolute paths
//! identically.

use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Decides whether a file should be indexed.
#[derive(Debug, Clone, Default)]
pub struct Filter {
    pub max_bytes: u64,
    /// Glob patterns, project-relative or base-name. `**` is supported.
    pub excludes: Vec<String>,
}

impl Filter {
    /// Sane defaults for CodeCorpus. Mirrors the Go
    /// `DefaultFilter()` byte-for-byte.
    pub fn default_values() -> Self {
        Self {
            max_bytes: 200 * 1024,
            excludes: [
                "vendor/**",
                "internal/assets/embedded/**",
                "docs/superpowers/specs/**",
                "*.lock",
                "*.sum",
                "package-lock.json",
                "*.png",
                "*.jpg",
                "*.gif",
                "*.ico",
                "*.pdf",
                "*.zip",
                "*.tar.gz",
            ]
            .iter()
            .map(|&s| s.into())
            .collect(),
        }
    }

    /// Reports whether `path` should be indexed. Path may be absolute
    /// or project-root-relative; glob matching runs against both the
    /// full path and the basename so base-only patterns like `*.sum`
    /// also catch vendored paths.
    pub fn accept(&self, path: &Path) -> bool {
        let path_str = path.to_string_lossy();
        for g in &self.excludes {
            if glob_match(g, &path_str) {
                return false;
            }
        }
        let Ok(info) = std::fs::metadata(path) else {
            return false;
        };
        if info.is_dir() {
            return false;
        }
        if self.max_bytes > 0 && info.len() > self.max_bytes {
            return false;
        }
        // Null-byte probe: read the first 1 KB and reject if any byte is 0.
        let Ok(mut fh) = File::open(path) else {
            return false;
        };
        let mut buf = [0u8; 1024];
        let n = fh.read(&mut buf).unwrap_or(0);
        if n > 0 && buf[..n].contains(&0) {
            return false;
        }
        true
    }
}

/// Glob match supporting `**` in addition to `glob::Pattern` syntax.
/// Matches against both the full path and the basename so base-only
/// patterns catch vendored paths.
pub fn glob_match(pattern: &str, path: &str) -> bool {
    if pattern.contains("**") {
        let mut parts = pattern.splitn(2, "**");
        let raw_prefix = parts.next().unwrap_or("");
        let raw_suffix = parts.next().unwrap_or("");
        let prefix = raw_prefix.trim_end_matches('/');
        let suffix = raw_suffix.trim_start_matches('/');

        if !prefix.is_empty()
            && !path.starts_with(&format!("{prefix}/"))
            && !path.contains(&format!("/{prefix}/"))
            && !has_segment(path, prefix)
        {
            return false;
        }
        if suffix.is_empty() {
            return true;
        }
        if let Ok(p) = glob::Pattern::new(suffix) {
            let base = std::path::Path::new(path)
                .file_name()
                .and_then(|os| os.to_str())
                .unwrap_or(path);
            return p.matches(base);
        }
        return false;
    }
    // No **: try full path, then basename.
    if let Ok(p) = glob::Pattern::new(pattern) {
        if p.matches(path) {
            return true;
        }
        let base = std::path::Path::new(path)
            .file_name()
            .and_then(|os| os.to_str())
            .unwrap_or(path);
        if p.matches(base) {
            return true;
        }
    }
    false
}

/// Reports whether `segment` appears as a single path component in `path`.
fn has_segment(path: &str, segment: &str) -> bool {
    path.split(std::path::MAIN_SEPARATOR)
        .any(|part| part == segment)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn skips_binary() {
        let dir = tempfile::tempdir().unwrap();
        let bin = dir.path().join("image.png");
        fs::write(&bin, [0x89, 0x50, 0x4e, 0x47, 0x00, 0x01, 0x02]).unwrap();
        assert!(!Filter::default_values().accept(&bin));
    }

    #[test]
    fn skips_oversized() {
        let dir = tempfile::tempdir().unwrap();
        let big = dir.path().join("big.txt");
        fs::write(&big, vec![b'x'; 300_000]).unwrap();
        assert!(!Filter::default_values().accept(&big));
    }

    #[test]
    fn honors_exclude_glob() {
        let f = Filter {
            excludes: vec!["vendor/**".into()],
            max_bytes: 1 << 30,
        };
        assert!(glob_match("vendor/**", "vendor/foo/bar.go"));
        assert!(glob_match("vendor/**", "vendor/foo.go"));
        // The accept() path short-circuits on the glob before stat, so we
        // don't need real files on disk to verify the exclusion semantics.
        assert!(!f.accept(std::path::Path::new("vendor/foo/bar.go")));
    }

    /// Regression: multi-segment exclude prefix must match even when the
    /// file is addressed by an absolute path. Before the Go fix, only
    /// "prefix/..." or single-segment prefixes matched.
    #[test]
    fn multi_segment_prefix_matches_absolute_path() {
        for p in [
            "docs/superpowers/specs/foo.md",
            "/workspace/docs/superpowers/specs/foo.md",
            "/home/dev/cspace/docs/superpowers/specs/nested/bar.md",
        ] {
            assert!(
                glob_match("docs/superpowers/specs/**", p),
                "should match {p}"
            );
        }
        assert!(!glob_match(
            "docs/superpowers/specs/**",
            "docs/src/content/foo.md"
        ));
        assert!(!glob_match(
            "docs/superpowers/specs/**",
            "/workspace/docs/src/content/foo.md"
        ));
    }

    #[test]
    fn accepts_normal_text_file() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("hello.go");
        fs::write(&p, "package main\n\nfn main() {}\n").unwrap();
        assert!(Filter::default_values().accept(&p));
    }

    #[test]
    fn honors_lock_and_sum_globs() {
        let f = Filter::default_values();
        let dir = tempfile::tempdir().unwrap();
        for name in ["go.sum", "package-lock.json"] {
            let p = dir.path().join(name);
            fs::write(&p, "{}").unwrap();
            assert!(!f.accept(&p), "{name} should be rejected");
        }
    }
}
