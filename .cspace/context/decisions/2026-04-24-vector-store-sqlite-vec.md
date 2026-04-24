# Vector store: stick with sqlite-vec; consider LanceDB after perf testing

Date: 2026-04-24
Status: terminal

## Decision

**Use [sqlite-vec](https://github.com/asg017/sqlite-vec) as the vector store for v0.1.** Reconsider only after real performance data from actual corpora, and in that scenario the upgrade target is **[LanceDB](https://github.com/lancedb/lancedb)**, not qdrant.

## Context

The Go version of `cspace search` used qdrant over HTTP in a Docker sidecar. The Rust port needed zero-install distribution via Homebrew ("brew install cspace-search, that's it"), so qdrant's separate-process architecture is a non-starter in any of its forms:

- `brew install qdrant` as a peer dep → violates the one-install constraint
- Auto-download qdrant binary + manage subprocess → ~1–2 days of lifecycle code (orphan cleanup, port collision, upgrade drift) that's hard to get right
- Embedded qdrant via `qdrant/rust-client` → not a thing; rust-client is a gRPC client to a running server. qdrant's internal crates (`segment`, `collection`) aren't stable library APIs

Performance doesn't force a switch at cspace-search's realistic corpus sizes (1k–50k chunks for most projects):

| Corpus | sqlite-vec query | qdrant query | Embed query (both) |
|---|---|---|---|
| 1k | ~1ms | <1ms | ~50ms |
| 10k | ~5–15ms | ~1–3ms | ~50ms |
| 100k | ~50–150ms | ~3–10ms | ~50ms |

At <50k vectors, the llama.cpp embedding step dominates latency; store choice is invisible to the user. sqlite-vec catches up to the embedder around ~100k vectors and falls off at ~500k+.

## Why LanceDB, not qdrant, if we ever upgrade

LanceDB is pure-Rust, embedded (single-file storage), HNSW-indexed, and designed for embedded RAG workflows. It's the closest thing to "qdrant performance inside a sqlite-vec distribution story":

- Single-binary distribution: ✅ (same as sqlite-vec)
- HNSW kNN: ✅ (unlike sqlite-vec's partial implementation)
- Rich payload filters: ✅ (sqlite-vec requires manual JSON1 queries)
- Concurrent readers: ✅
- Maturity: middle (younger than qdrant, older than sqlite-vec)

qdrant is a better vector store in absolute terms but its distribution model is wrong for us. LanceDB trades a little maturity for keeping our install story intact.

## Conditions that would trigger the LanceDB evaluation

Revisit this decision only if one of these signals fires:

1. **Benchmarked >50k vectors showing user-visible latency regression.** Run the micro-benchmark first (see follow-up below); only act on measured data, not vibes.
2. **Concurrency story becomes a bottleneck.** Specifically, if we add autonomous-agent re-indexing that needs to interleave with live MCP queries, sqlite-vec's single-writer model will show.
3. **Payload-filter complexity grows.** `path_filter` / `kind_filter` in Phase 6 are fine on sqlite-vec. If Phase 7+ adds multi-field filters, numeric range predicates, or faceted search, the manual-JSON1 approach gets painful.

## Re-evaluation path

No scheduled work. If a trigger condition above fires in real use, spike
LanceDB as a drop-in alternative `Upserter` + `Searcher` implementation.
The Phase 3 trait split makes this bounded: a second impl in
`src/index/lancedb.rs`, a runtime pick based on config, done. No
surface-level changes upstream.

## What we are NOT doing

- **Not** adopting qdrant in any form. Distribution model is wrong. Settled.
- **Not** pre-benchmarking against synthetic corpora. The decision is
  grounded in distribution constraints; benchmarks would inform a
  re-evaluation, not the current call.
