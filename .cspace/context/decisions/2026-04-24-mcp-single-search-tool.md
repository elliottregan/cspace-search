# MCP server: single `search` tool instead of per-corpus dynamic tools

Date: 2026-04-24
Status: terminal

## Decision

Phase 6 ships a single `search` tool that takes `corpus` as a required
parameter, rather than synthesizing one MCP tool per enabled corpus
(`search_code`, `search_commits`, `search_context`, `search_issues`).
`search_status` is the second (and only other) tool.

## Context

The original design called for dynamic per-corpus tools: at server
startup, read the config, and emit a `search_<corpus>` tool for every
enabled corpus. Clients would then see a dropdown of corpus-specific
tools with pre-narrowed schemas.

rmcp 1.5's tool registration is macro-driven at compile time
(`#[tool_router] impl`, `#[tool(description = ...)]` on methods).
Registering N tools at runtime requires bypassing the macro and
hand-rolling a `ServerHandler` impl that:

- Implements `list_tools` to iterate the config and emit `Tool`
  descriptors with a dynamically-generated input schema per corpus.
- Implements `call_tool` with a name-prefix match
  (`tool.name.strip_prefix("search_")`) to dispatch to the same
  underlying search code.

That's meaningful ceremony (maybe a day of wiring + tests) for
effectively zero behaviour change: the set of corpora is tiny and
stable (code, commits, context, issues), and clients that want a
per-corpus shortcut can wrap the single tool themselves.

## What this gives up

- A client surface where "search code" and "search commits" are
  visually separate tool names.
- Per-corpus input schemas that could specialize — e.g. forbidding
  `kind_filter` on the commits corpus where it's meaningless.

Neither is load-bearing for v0.1. Every MCP client we've seen surfaces
tools as a flat list of callable functions; clients that lean on
name-based autocomplete are rare.

### 2026-04-24 follow-up: corpus advertising

Both original concessions are now addressed without going back to
dynamic tools. `ServerHandler::list_tools` is overridden to read the
runnable corpus set at call time and inject per-corpus specialisation
into the `search` tool:

- `properties.corpus.enum` — valid corpus ids
- `properties.kind_filter.description` — the kind vocabulary each
  corpus emits, formatted as `corpus=kind1|kind2; …`
- Top-level tool description — summarises corpora, their kinds, and
  which corpora don't support `path_filter` / `include_preview`
  (commits is the current example — `path` is a SHA, not a file path)

The vocabulary comes from two new trait methods on `Corpus`:

- `kinds() -> Vec<String>` — the kind values this corpus emits,
  sorted and deduped. `CommitCorpus` returns `["commit"]`;
  `FileCorpus` unions its path_groups' kinds with the default.
- `supports_paths() -> bool` — true when the `path` field is a
  file path (amenable to glob filtering and line-range preview).
  `CommitCorpus` returns false; `FileCorpus` defaults to true.

Clients that render JSON Schema constraints get an enum dropdown for
`corpus` and a contextual description for `kind_filter`. Clients that
only render the tool description see a structured rundown. The
ceremony is ~80 lines in `commands/mcp.rs` plus two ~10-line trait
methods, not a per-corpus tool synthesis with its own dispatch.

## Re-evaluation trigger

Flip back to dynamic per-corpus tools if:

1. An MCP client we care about starts ranking or routing based on tool
   name and a human reports `search` feels buried.
2. Corpus-specific input schema constraints become valuable (e.g. a
   new corpus type ships that has its own unique filter fields).

At that point, hand-roll `ServerHandler` and drop the macro approach
for this command only.
