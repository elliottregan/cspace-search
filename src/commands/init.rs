use crate::config;
use crate::embed::FakeEmbedder;
use crate::index::{self, sqlite::SqliteUpserter, RunConfig};
use clap::Parser;
use std::path::PathBuf;

/// Build or refresh the search index for the current project.
///
/// Currently wires the indexer against the fake embedder; the real
/// Jina v5 embedder lands once the ONNX runtime pick stabilizes (see
/// PORTING.md Phase 4b). The pipeline shape — enumerate → hash-skip →
/// embed → upsert → evict orphans — matches the Go version so swapping
/// the embedder later is a one-line change here.
#[derive(Parser, Debug)]
pub struct Args {
    /// Project root. Defaults to the nearest ancestor directory with a `.git/`.
    #[arg(long)]
    pub root: Option<PathBuf>,

    /// Which corpus to index. Repeatable; "all" expands to every
    /// enabled corpus in the resolved config.
    #[arg(long, value_name = "ID", default_values_t = ["all".to_string()])]
    pub corpus: Vec<String>,

    /// Suppress per-batch progress output.
    #[arg(long)]
    pub quiet: bool,

    /// Embedding dimensionality. Must match the real embedder dim
    /// once Phase 4b lands (Jina v5 nano is 768); kept as a flag for
    /// the interim fake-embedder path so tests and ad-hoc runs can
    /// exercise small dims without stomping a 768-dim store.
    #[arg(long, default_value_t = 768)]
    pub dim: usize,
}

pub fn run(args: Args) -> anyhow::Result<()> {
    let root = args.root.map(Ok).unwrap_or_else(find_project_root)?;
    let cfg = config::load(&root)?;

    let corpora: Vec<String> = if args.corpus.iter().any(|c| c == "all") {
        cfg.corpora
            .iter()
            .filter(|(_, cc)| cc.enabled)
            .map(|(id, _)| id.clone())
            .collect()
    } else {
        args.corpus.clone()
    };

    if corpora.is_empty() {
        eprintln!("No corpora enabled. Set `enabled: true` in {}/search.yaml and enable the corpora you want indexed.", root.display());
        return Ok(());
    }

    let db_path = index_db_path(&root)?;
    let upserter = SqliteUpserter::open(&db_path)?;
    let embedder = FakeEmbedder::new(args.dim);

    for id in &corpora {
        // Build the corpus via the runtime gate so a disabled corpus
        // produces the canonical error surface.
        let runtime = match config::runtime::build_with_config(&root, id, cfg.clone()) {
            Ok(rt) => rt,
            Err(e) => {
                eprintln!("skipping {id}: {e}");
                continue;
            }
        };

        // Bind the progress closure to a named local so it outlives
        // the RunConfig struct literal (taking `&<closure>` of a
        // temporary drops it at the end of the statement).
        let progress = |done: usize, total: usize| {
            eprint!("\r  {id}: {done}/{total}");
            if done == total {
                eprintln!();
            }
        };
        let stats = index::run(RunConfig {
            corpus: runtime.corpus.as_ref(),
            embedder: &embedder,
            upserter: &upserter,
            project_root: &root,
            batch_size: 0,
            progress: if args.quiet { None } else { Some(&progress) },
        })?;

        println!(
            "{id}: enumerated={} embedded={} orphans_deleted={}",
            stats.enumerated, stats.embedded, stats.orphans_deleted
        );
    }

    eprintln!();
    eprintln!(
        "note: running with FakeEmbedder; vectors are deterministic but not semantic. \
         Phase 4b swaps in Jina v5."
    );
    Ok(())
}

/// Walk up from `cwd` to find the nearest `.git/` directory, then
/// return that directory as the project root.
fn find_project_root() -> anyhow::Result<PathBuf> {
    let mut cur = std::env::current_dir()?;
    loop {
        if cur.join(".git").exists() {
            return Ok(cur);
        }
        if !cur.pop() {
            break;
        }
    }
    // Fall back to cwd; most callers pass --root explicitly anyway.
    Ok(std::env::current_dir()?)
}

/// Index sqlite file path. One per project, keyed by project hash so
/// distinct clones of the same repo don't collide under
/// `~/.cspace-search/`.
fn index_db_path(project_root: &std::path::Path) -> anyhow::Result<PathBuf> {
    let hash = crate::corpus::project_hash(project_root);
    let home = dirs_home()?;
    Ok(home.join(".cspace-search").join(format!("{hash}.db")))
}

fn dirs_home() -> anyhow::Result<PathBuf> {
    // std doesn't give us a real cross-platform home dir, but for our
    // supported platforms (macOS + Linux) $HOME is sufficient.
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("HOME is not set"))
}
