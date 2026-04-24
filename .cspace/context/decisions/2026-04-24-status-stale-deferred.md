# `search_status.stale` always null in v0.1

Date: 2026-04-24
Status: terminal

## Decision

`search_status` returns `stale: null` for every invocation in v0.1.
The field is reserved in the envelope so a future version can populate
it without a breaking schema change, but detection itself is not
implemented.

## Context

A useful `stale` signal would fire when the index lags the underlying
corpus — e.g. files have been edited but `cspace-search init` hasn't
been re-run. The honest way to detect this is to re-enumerate the
corpus and compare content hashes against `existing_points`, which is
what `init` already does.

Building that into `search_status` has two problems:

1. **It makes a "status" call an O(corpus) operation.** Status is
   meant to be a fast, read-only tool a client can poll; staleness
   detection would make each call re-walk every tracked path.
2. **It partially duplicates `init`.** The indexer already knows how
   many records were embedded vs. skipped — exposing that count at the
   end of each run is a better API than recomputing it on demand.

## What we DO surface (2026-04-24 follow-up)

`search_status` per corpus returns:

- `row_count` — vectors currently stored
- `last_indexed_at` — unix seconds of the most recent write (create,
  upsert, or delete) to the collection
- `fingerprint_matches` — true when the stored fingerprint matches
  what the currently-loaded embedder would produce; false means a
  re-`init` would drop and rebuild the vectors (model swap detected)
- `stored_fingerprint` — the fingerprint that produced the existing
  vectors, for debugging mismatches

Plus top-level `index.db_size_bytes` and `index.db_path`.

Clients that want a heuristic "is the index stale?" can combine
`fingerprint_matches` (authoritative — false means definitely stale)
with `last_indexed_at` recency against whatever change signal they
track (e.g. git HEAD, file mtimes). The point is: we give enough
shards for a sensible client heuristic without making `status` do
O(corpus) work.

## What consumers should do instead

Callers that want authoritative stale signal:

1. Run `cspace-search init` (it's fast on a warm index — hash-skip
   means only truly changed records get re-embedded).
2. Read the stats printed at the end: `enumerated`, `embedded`,
   `orphans_deleted`. Any non-zero `embedded` or `orphans_deleted`
   means the index wasn't current.

## Re-evaluation trigger

Add real stale detection if a client meaningfully needs to know the
answer without running `init` first. Possible cheap signals:

- Compare sqlite file mtime to the newest tracked file's mtime
  (fast, but false-positive prone — ignores unchanged file rewrites).
- Compare git HEAD at last `init` time to current HEAD (needs a
  metadata row; reasonable for the `code` corpus, less so for the
  others).
