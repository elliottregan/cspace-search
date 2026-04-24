//! cspace-search — semantic search over commits, code, and project context.
//!
//! Port of the Go `cspace search` subcommand into a standalone Rust binary.
//! The goal is a single `brew install`-able tool with no Docker dependency
//! for the core index/search/mcp paths. `clusters` is the one feature that
//! still may want an external dim-reduction service; see `mod commands::clusters`.

use clap::{Parser, Subcommand};

mod commands;
mod config;
mod corpus;
mod embed;
mod index;

/// cspace-search: local-first semantic search for commits, code, and context.
#[derive(Parser, Debug)]
#[command(name = "cspace-search", version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Build or refresh the search index for the current project.
    Init(commands::init::Args),
    /// Semantic search across the indexed corpus.
    Search(commands::search::Args),
    /// Print index staleness + size + model version.
    Status(commands::status::Args),
    /// Dim-reduce + cluster embeddings for visualization.
    Clusters(commands::clusters::Args),
    /// Run the stdio MCP server exposing search tools.
    Mcp(commands::mcp::Args),
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();
    match cli.command {
        Command::Init(args) => commands::init::run(args),
        Command::Search(args) => commands::search::run(args),
        Command::Status(args) => commands::status::run(args),
        Command::Clusters(args) => commands::clusters::run(args),
        Command::Mcp(args) => commands::mcp::run(args),
    }
}
