use clap::Parser;

/// Dim-reduce + cluster embeddings for visualization.
///
/// This command requires a reduce-api service (PaCMAP / LocalMAP) because no
/// production-quality Rust implementation exists. Two supported modes:
///   1. Point at a running mindthemath/reduce-api instance via --reduce-url.
///   2. Run the bundled docker-compose.yml alongside cspace-search; the
///      command auto-detects it and talks to localhost.
///
/// Core init/search/status do NOT require this service.
#[derive(Parser, Debug)]
pub struct Args {
    /// Project root.
    #[arg(long)]
    pub root: Option<std::path::PathBuf>,

    /// URL of the reduce-api (PaCMAP) service.
    #[arg(long, default_value = "http://localhost:8000")]
    pub reduce_url: String,

    /// URL of the HDBSCAN clustering service.
    #[arg(long, default_value = "http://localhost:8090")]
    pub hdbscan_url: String,
}

pub fn run(_args: Args) -> anyhow::Result<()> {
    anyhow::bail!("clusters: not implemented yet — see PORTING.md phase list")
}
