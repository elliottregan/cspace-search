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

## Re-evaluation trigger

Flip back to dynamic per-corpus tools if:

1. An MCP client we care about starts ranking or routing based on tool
   name and a human reports `search` feels buried.
2. Corpus-specific input schema constraints become valuable (e.g. a
   new corpus type ships that has its own unique filter fields).

At that point, hand-roll `ServerHandler` and drop the macro approach
for this command only.
