use clap::Parser;

/// Build or refresh the search index for the current project.
#[derive(Parser, Debug)]
pub struct Args {
    /// Project root. Defaults to the nearest ancestor directory with a `.git/`.
    #[arg(long)]
    pub root: Option<std::path::PathBuf>,

    /// Suppress per-batch progress output.
    #[arg(long)]
    pub quiet: bool,

    /// Re-index every corpus entry, ignoring the content-hash skip cache.
    #[arg(long)]
    pub force: bool,
}

pub fn run(_args: Args) -> anyhow::Result<()> {
    anyhow::bail!("init: not implemented yet — see PORTING.md phase list")
}
