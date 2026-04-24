use clap::Parser;

/// Run the stdio MCP server exposing search tools (search_code,
/// search_commits, search_context, search_issues, list_clusters).
#[derive(Parser, Debug)]
pub struct Args {
    /// Project root. Defaults to the nearest ancestor directory with a `.git/`.
    #[arg(long)]
    pub root: Option<std::path::PathBuf>,
}

pub fn run(_args: Args) -> anyhow::Result<()> {
    anyhow::bail!("mcp: not implemented yet — see PORTING.md phase list")
}
