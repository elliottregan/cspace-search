use crate::config;
use crate::embed::llama::{LlamaEmbedder, DEFAULT_DIM};
use crate::embed::{Embedder, FakeEmbedder};
use crate::index::{self, sqlite::SqliteUpserter, RunConfig};
use crate::util;
use clap::Parser;
use std::path::PathBuf;

/// Build or refresh the search index for the current project.
///
/// Uses the llama.cpp-backed Jina v5 nano retrieval embedder by
/// default; pass `--fake-embedder` to substitute a deterministic
/// sha-seeded fake (small, no model download, useful for CI and for
/// exercising the pipeline without inference cost). The model +
/// tokenizer download to `~/.cache/huggingface/hub/` on first use.
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

    /// Use the deterministic FakeEmbedder instead of loading the
    /// real Jina v5 model. Produces unit vectors keyed by sha256 of
    /// the input text — fine for exercising the pipeline, useless
    /// for semantic search. `--dim` applies only in fake mode.
    #[arg(long)]
    pub fake_embedder: bool,

    /// Embedding dimensionality. Ignored unless `--fake-embedder` is
    /// set; the real embedder always reports its own native dim
    /// (768 for Jina v5 nano).
    #[arg(long, default_value_t = DEFAULT_DIM)]
    pub dim: usize,
}

pub fn run(args: Args) -> anyhow::Result<()> {
    let root = args.root.map(Ok).unwrap_or_else(util::find_project_root)?;
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

    let db_path = util::index_db_path(&root)?;
    let upserter = SqliteUpserter::open(&db_path)?;
    // Boxed to a trait object so both embedders share a codepath below.
    let embedder: Box<dyn Embedder> = if args.fake_embedder {
        Box::new(FakeEmbedder::new(args.dim))
    } else {
        eprintln!("Loading Jina v5 nano retrieval (first use downloads ~80MB)...");
        Box::new(LlamaEmbedder::jina_v5_nano_retrieval()?)
    };

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
            embedder: embedder.as_ref(),
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

    if args.fake_embedder {
        eprintln!();
        eprintln!(
            "note: --fake-embedder in use; vectors are deterministic but not semantic. \
             Drop the flag to index with Jina v5."
        );
    }
    Ok(())
}
