# cspace-search

Local-first semantic search for commits, code, and project context. A standalone
port of the `cspace search` subcommand from [cspace](https://github.com/elliottregan/cspace),
rewritten in Rust for a single bundled binary with no Docker dependency on the
core paths.

Status: **early port, not yet functional.** Subcommands parse but every `run()`
returns an error directing you here. See [PORTING.md](PORTING.md) for the phase
list.

## Why a rewrite

cspace's search was Go + four Docker services (llama-server, qdrant, reduce-api,
hdbscan-api). Great for a devcontainer tool that already ran Docker; heavy for
a search binary that should install via Homebrew. The rewrite collapses the
core embed + vector store into the binary itself:

- **Embedding**: [Candle](https://github.com/huggingface/candle) runs the Jina v5
  nano model natively on Metal / CPU. No llama.cpp C++ shipping, no Ollama
  dependency, no server process.
- **Vector DB**: [sqlite-vec](https://github.com/asg017/sqlite-vec). Embedded.
  Data in `~/.cspace-search/<project-hash>.db`.
- **Corpus**: commits via [git2](https://github.com/rust-lang/git2-rs), code via
  [ignore](https://github.com/BurntSushi/ripgrep/tree/master/crates/ignore),
  context via local files, issues via `gh api`.
- **MCP server**: stdio via [rmcp](https://github.com/modelcontextprotocol/rust-sdk).

The one holdout is `clusters`, which still needs PaCMAP dim-reduction and HDBSCAN
— neither has a production-quality Rust implementation. It's kept behind an
optional external service (`reduce-api`, `hdbscan-api`) that users can run via a
`docker-compose.yml` bundled in this repo if they want the feature, or skip.

## Install

Not yet published. For now:

```sh
git clone https://github.com/elliottregan/cspace-search
cd cspace-search
cargo install --path .
```

Will ship via Homebrew tap once Phase C lands.

## Usage

```sh
cspace-search init                    # build the index
cspace-search "authentication flow"   # semantic query
cspace-search status                  # staleness + size + model version
cspace-search mcp                     # stdio MCP server
cspace-search clusters                # (requires optional docker-compose)
```

## Layout

```
src/
├── main.rs            # CLI dispatch
├── commands/          # one module per subcommand
├── config/            # search.yaml loading (parity with the Go version)
├── corpus/            # commit / code / context / issues walkers
├── embed/             # candle-based embedder
├── index/             # sqlite-vec storage + content-hash skip cache
├── query/             # search + ranking
└── mcp/               # MCP server
```

## License

Same as cspace: [PolyForm Perimeter 1.0.0](LICENSE).
