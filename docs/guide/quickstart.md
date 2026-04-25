# Quick start

Index your project, run a query, and (optionally) expose search to an
agent over MCP. Five minutes from cold install to first hit.

## 1. Install

```sh
brew install elliottregan/cspace/cspace-search
```

The Homebrew tap is not yet published — until then, build from source
following [Installation](./installation).

## 2. Index a project

From any directory inside a git repo:

```sh
cspace-search init
```

The first run downloads the Jina v5 nano retrieval model (~80 MB) into
`~/.cache/huggingface/hub/`. Subsequent runs reuse the cached model and
the global [embedding cache](../architecture/what-happens-during-a-search#three-databases-three-roles).

By default `init` indexes every corpus enabled in `search.yaml`. Pass
`--corpus <id>` to limit; `--corpus all` is the explicit form of the
default. Disable the global cache for a clean rebuild with
`--no-embed-cache`.

## 3. Search

```sh
cspace-search search "how do we handle merges?"
```

Output (human form):

```
0.812  .cspace/context/principles.md:14-28  (context)
0.689  CONTRIBUTING.md:42-58                 (context)
0.634  .cspace/context/decisions/...         (context)
```

Pass `--corpus commits` (or `code`, `context`, …) to switch corpora,
`--top-k N` to change the result count, or `--json` for a
machine-readable envelope.

## 4. Expose to an agent (MCP)

```sh
cspace-search mcp
```

This is a stdio JSON-RPC server speaking the
[Model Context Protocol](https://modelcontextprotocol.io). It exposes
two tools:

- **`search`** — kNN over a named corpus, with `limit`, `path_filter`,
  `kind_filter`, and `include_preview` shaping. The tool's input schema
  advertises the runnable corpus set as an enum, and the per-corpus
  `kind` vocabulary in the `kind_filter` description.
- **`search_status`** — per-corpus row counts, last-indexed timestamps,
  fingerprint match state, and overall db size. Use this to detect
  when a re-index would help.

Register it with Claude Code:

```sh
claude mcp add --scope user cspace-search -- cspace-search mcp --root /path/to/project
```

Or any MCP client that speaks stdio.

## What just happened

The detailed walk-through is in
[What actually happens during a search](../architecture/what-happens-during-a-search).
The short version: `init` chunked your project's files, ran each chunk
through the embedding model, and stored the resulting 768-dim vectors
in a sqlite database keyed by your project's path hash. `search`
embedded your query the same way and asked sqlite-vec for the rows
whose embeddings sit closest to the query in that 768-dim space.

## Where things live

- **Index database** — `~/.cspace-search/<project_hash>.db`
- **Embedding cache** — `~/.cspace-search/embed-cache.db`
- **Model file** — `~/.cache/huggingface/hub/...v5-nano-Q8_0.gguf`

Wiping any or all of those is safe. The cache and the model
re-download / re-derive themselves; the index is one `cspace-search
init` away.
