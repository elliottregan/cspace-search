use clap::Parser;

/// Semantic search across the indexed corpus.
#[derive(Parser, Debug)]
pub struct Args {
    /// Query text.
    pub query: String,

    /// Project root. Defaults to the nearest ancestor directory with a `.git/`.
    #[arg(long)]
    pub root: Option<std::path::PathBuf>,

    /// Restrict to a single corpus (commits, code, context, issues).
    #[arg(long)]
    pub corpus: Option<String>,

    /// Number of hits to return.
    #[arg(long, default_value_t = 10)]
    pub top_k: usize,
}

pub fn run(_args: Args) -> anyhow::Result<()> {
    anyhow::bail!("search: not implemented yet — see PORTING.md phase list")
}
