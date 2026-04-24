use clap::Parser;

/// Print index staleness + size + model version.
#[derive(Parser, Debug)]
pub struct Args {
    /// Project root. Defaults to the nearest ancestor directory with a `.git/`.
    #[arg(long)]
    pub root: Option<std::path::PathBuf>,

    /// Output machine-readable JSON.
    #[arg(long)]
    pub json: bool,
}

pub fn run(_args: Args) -> anyhow::Result<()> {
    anyhow::bail!("status: not implemented yet — see PORTING.md phase list")
}
