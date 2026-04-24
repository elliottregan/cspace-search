//! Shared helpers used by multiple subcommands. Nothing here is
//! conceptually new — these are just small functions that were
//! duplicated between `commands/init.rs` and `commands/search.rs`
//! before the cleanup pass.

use std::path::PathBuf;

/// Walk up from the current working directory until we hit a `.git/`
/// entry, and return that directory as the project root. Falls back
/// to `cwd` if no `.git/` is found anywhere on the walk up — most
/// commands take `--root` explicitly anyway.
pub fn find_project_root() -> anyhow::Result<PathBuf> {
    let mut cur = std::env::current_dir()?;
    loop {
        if cur.join(".git").exists() {
            return Ok(cur);
        }
        if !cur.pop() {
            break;
        }
    }
    Ok(std::env::current_dir()?)
}

/// Path to the per-project sqlite index file. Keyed by project-root
/// hash so distinct clones of the same repo don't collide under
/// `~/.cspace-search/`.
pub fn index_db_path(project_root: &std::path::Path) -> anyhow::Result<PathBuf> {
    let hash = crate::corpus::project_hash(project_root);
    Ok(home_dir()?
        .join(".cspace-search")
        .join(format!("{hash}.db")))
}

/// The user's home directory, from `$HOME`. macOS + Linux only —
/// Windows support is out of scope for v0.1 and would use
/// `%USERPROFILE%` plus a different default config location anyway.
pub fn home_dir() -> anyhow::Result<PathBuf> {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("HOME is not set"))
}
