use crate::config;
use crate::embed::llama::LlamaEmbedder;
use crate::embed::{Embedder, FakeEmbedder};
use crate::index::sqlite::SqliteUpserter;
use crate::query::{self, RunConfig};
use clap::Parser;
use std::path::PathBuf;

/// Semantic search across the indexed corpus.
#[derive(Parser, Debug)]
pub struct Args {
    /// Query text.
    pub query: String,

    /// Project root. Defaults to the nearest ancestor directory with a `.git/`.
    #[arg(long)]
    pub root: Option<PathBuf>,

    /// Which corpus to search. Defaults to `code`, matching the
    /// common case; `context`, `commits`, and `issues` are also
    /// valid once they've been indexed.
    #[arg(long, default_value = "code")]
    pub corpus: String,

    /// Number of hits to return. Clamped to `[1, 50]`.
    #[arg(long, default_value_t = 10)]
    pub top_k: usize,

    /// Output machine-readable JSON instead of the human table.
    #[arg(long)]
    pub json: bool,

    /// Use the deterministic FakeEmbedder instead of loading Jina v5.
    /// Query results will only match documents indexed with the same
    /// flag.
    #[arg(long)]
    pub fake_embedder: bool,

    /// Embedding dim for `--fake-embedder`. Ignored otherwise.
    #[arg(long, default_value_t = crate::embed::llama::DEFAULT_DIM)]
    pub dim: usize,
}

pub fn run(args: Args) -> anyhow::Result<()> {
    let root = args.root.map(Ok).unwrap_or_else(find_project_root)?;
    let cfg = config::load(&root)?;

    let runtime = config::runtime::build_with_config(&root, &args.corpus, cfg)?;

    let embedder: Box<dyn Embedder> = if args.fake_embedder {
        Box::new(FakeEmbedder::new(args.dim))
    } else {
        Box::new(LlamaEmbedder::jina_v5_nano_retrieval()?)
    };

    let db_path = index_db_path(&root)?;
    let searcher = SqliteUpserter::open(&db_path)?;

    let envelope = query::run(RunConfig {
        corpus: runtime.corpus.as_ref(),
        embedder: embedder.as_ref(),
        searcher: &searcher,
        project_root: &root,
        query: &args.query,
        top_k: args.top_k,
        with_cluster: false,
    })?;

    if args.json {
        println!("{}", serde_json::to_string_pretty(&envelope)?);
    } else {
        if !envelope.warning.is_empty() {
            eprintln!("warning: {}", envelope.warning);
        }
        if envelope.results.is_empty() {
            eprintln!("no results in corpus {:?}", envelope.corpus);
        }
        for h in &envelope.results {
            let range = if h.line_start > 0 {
                format!(":{}-{}", h.line_start, h.line_end)
            } else {
                String::new()
            };
            println!("{:>6.3}  {}{}  ({})", h.score, h.path, range, h.kind);
        }
    }
    Ok(())
}

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
    Ok(std::env::current_dir()?)
}

fn index_db_path(project_root: &std::path::Path) -> anyhow::Result<PathBuf> {
    let hash = crate::corpus::project_hash(project_root);
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("HOME is not set"))?;
    Ok(home.join(".cspace-search").join(format!("{hash}.db")))
}
