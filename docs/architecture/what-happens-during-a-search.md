# What actually happens during a search

When an MCP client (or the CLI) runs
`search("how do we handle merges?", corpus="context")`, several things
have to happen between the user's text and the ranked list of hits
that comes back. This article traces that path step by step,
identifies which file on disk does what, and explains why the
architecture looks the way it does.

## The big picture

A search is a function of three inputs:

1. **The user's query string** — natural language text.
2. **The embedding model** — a fixed function that maps text to a
   768-dimensional unit vector. cspace-search uses Jina v5 nano
   retrieval by default.
3. **The index** — a sqlite database holding one vector per chunk of
   indexed content, plus enough metadata to make a hit useful (path,
   line range, kind, content hash).

The output is a ranked list of `Hit`s, each with a similarity score,
a path, and (optionally) a line range or inline text snippet.

## End-to-end trace

```
1. SearchServer::do_search                                    (commands/mcp.rs)
   │
2. config::runtime::build_with_config(root, "context", cfg)   (config/runtime.rs)
   │   gates the corpus through search.yaml's enabled flags,
   │   instantiates the FileCorpus for "context"
   │
3. query::run(RunConfig { corpus, embedder, searcher, ... })  (query/mod.rs)
   │
   ├─ embedder.embed_query("how do we handle merges?")        (embed/llama.rs)
   │   │
   │   ├─ Cache check: embed-cache.db                         (embed/cache.rs)
   │   │   SELECT vector FROM embed_cache
   │   │     WHERE fingerprint = '<fp>|query'
   │   │       AND text_hash   = sha256("how do we handle merges?")
   │   │   Hit → return cached 768-dim vector. Done in <1 ms.
   │   │
   │   └─ Miss → run llama.cpp:
   │       ├─ Prepend "Query: " prefix (Jina v5 retrieval convention)
   │       ├─ Tokenize via the model's vocab → ~10 token IDs
   │       ├─ Forward pass through the 80 MB GGUF (~50 ms on M-series)
   │       ├─ Mean-pool the contextualized token embeddings → 768 floats
   │       ├─ L2-normalize → unit vector
   │       └─ Store in embed-cache.db, return
   │
   ├─ corpus.collection(project_root) → "context-83af291c"    (corpus/file.rs)
   │   This is the vec0 table name in the index db.
   │
   ├─ searcher.search("context-83af291c", &q_vec, top_k * 3)  (index/sqlite.rs)
   │   │
   │   └─ Run against the index db's vec0 table:
   │
   │     SELECT rowid, distance, payload
   │     FROM   "context-83af291c"
   │     WHERE  embedding MATCH ?<q_vec_as_blob>
   │     ORDER  BY distance
   │     LIMIT  30
   │
   │     What vec0 is doing inside MATCH:
   │     ┌────────────────────────────────────────────┐
   │     │ for each of N rows:                        │
   │     │   load row.embedding (768 LE f32 = 3072 B) │
   │     │   d = L2(row.embedding, q_vec)             │
   │     │   maintain top-30 min-heap by distance     │
   │     │ emit the heap, sorted ascending            │
   │     └────────────────────────────────────────────┘
   │     This is brute-force kNN. No ANN index, no HNSW
   │     (yet — sqlite-vec is still exact-search-only).
   │     Fine up to ~50 K rows; visibly slow past 100 K.
   │
   ├─ For each returned row:
   │   score = 1.0 / (1.0 + distance)
   │     (monotonic remap of L2 → [0, 1] for human readability;
   │      the ranking is identical to L2 ascending. Since all
   │      vectors are L2-normalized, L2 ordering and cosine
   │      ordering are also identical.)
   │
   ├─ hit_from_raw(raw)  (query/mod.rs)
   │   Pulls path/kind/line_start/line_end/content_hash/cluster_id
   │   from payload into typed Hit fields. Forwards everything
   │   else (commit hash, finding status, eventually inline text)
   │   into Hit.extra.
   │
   ├─ dedupe_by_path  (query/mod.rs)
   │   Collapse multiple chunks of the same path, keep the
   │   highest-scoring one per path. (The OVERFETCH constant
   │   is why we asked for 30 instead of 10 — gives the dedup
   │   slack so it doesn't return short.)
   │
   └─ truncate to top_k (default 10)

→ Vec<Hit> returned to the agent through the MCP JSON-RPC transport.
```

## Three databases, three roles

cspace-search uses three sqlite files at runtime. Only one of them is
touched during the kNN search itself; the other two are read once at
startup or consulted lazily.

```
                              SEARCH-TIME ROLES

   ~/.cache/huggingface/                     read at server start
   └─ models--jinaai--…/v5-nano-Q8_0.gguf    (load model into RAM)
                                             not touched per query

   ~/.cspace-search/embed-cache.db           consulted BEFORE the model
   └─ embed_cache(fp, text_hash) → vector    (skip llama.cpp if hit)
                                             tiny: text_hash + 3 KB blob

   ~/.cspace-search/<project_hash>.db        the actual search DB
   ├─ context-<hash>     (vec0)              ← MATCH runs here
   ├─ code-<hash>        (vec0)              ← or here
   ├─ commits-<hash>     (vec0)              ← or here
   └─ collection_meta                        (read for fingerprint check)
```

### `~/.cache/huggingface/hub/.../v5-nano-Q8_0.gguf`

The embedding model itself — about 80 MB of binary weights in GGUF
format. Not a database in the SQL sense (it's not even sqlite) but it
lives in the same conceptual place: cached state on disk that the
search process reads at startup. cspace-search downloads it on first
use via the `hf-hub` crate; thereafter it's a plain file read.

Host-scoped: every project on the same machine shares this file.

### `~/.cspace-search/embed-cache.db`

Pure optimization. Memoizes
`(model_fingerprint, sha256(text)) → vector` so re-running `init` on a
mostly-unchanged corpus rarely has to compute a fresh embedding.
Wiping this file is safe — it just means the next big rebuild costs
CPU.

Query-side embeddings get cached too, under a separate fingerprint
key with a `|query` suffix. Jina retrieval models prepend a different
prefix (`"Query: "` vs `"Document: "`) at query time than at index
time, so the query and doc embeddings of the same string produce
different vectors. The cache key keeps them separate.

### `~/.cspace-search/<project_hash>.db`

The actual search database. Inside this single file there's:

- **One vec0 virtual table per corpus** — `code-<hash>`,
  `commits-<hash>`, `context-<hash>`, etc. Each row holds a
  768-dimensional vector, a content hash for change detection, and a
  JSON payload with the chunk's metadata.
- **One regular table `collection_meta`** — tracks per-corpus
  fingerprint, dim, and last-indexed timestamp. Lets the indexer
  detect a model swap and trigger drop+rebuild.

vec0 is the only table type that knows how to do
`WHERE embedding MATCH ?`. The rest is plain sqlite.

## What `MATCH` actually does

The `embedding MATCH ?` clause is where sqlite-vec's vec0 virtual-
table module takes over from regular sqlite. Each row's `embedding`
column stores 768 little-endian f32 values (3072 bytes) packed into a
BLOB. `MATCH` decodes the query blob and runs:

```
for each of N rows:
  load row.embedding (3072 B)
  d = L2(row.embedding, q_vec)
  maintain top-K min-heap by distance
emit the heap, sorted ascending by distance
```

This is brute-force kNN. There's no approximate nearest-neighbour
index — no HNSW, no IVF, no quantization. Every query scans every
row's vector. sqlite-vec is exact-search-only as of this writing.

### Why not ANN?

ANN indexes (HNSW, IVF) trade exact recall for sublinear query time.
They're the right choice once a corpus is large enough that linear
scan dominates query latency.

For cspace-search's typical scale — a single project's code (1 K to
50 K chunks) — linear scan stays fast (under ~30 ms) and the embedding
step (~50 ms cold, <1 ms cached) dominates anyway. The first knob to
reach for at larger scales is the vector store, not the index
structure: switching to LanceDB (HNSW out of the box, single-binary
distribution, pure Rust) is a planned upgrade if real-world corpora
ever push past ~100 K chunks.

See the [vector store decision
record](https://github.com/elliottregan/cspace-search/blob/main/.cspace/context/decisions/2026-04-24-vector-store-sqlite-vec.md)
for the full reasoning.

## Distance, similarity, and score

The model produces unit-normalized embeddings — every vector has L2
norm 1. For unit vectors, two metrics that look different are
mathematically equivalent in their ordering:

- **L2 distance**: `sqrt(2 - 2·cos)` where `cos` is the cosine
  similarity.
- **Cosine similarity**: the dot product.

Both produce identical rankings — L2 distance is a monotonically
decreasing function of cosine similarity. sqlite-vec computes L2
because that's what vec0 implements; cspace-search converts it to a
more readable score:

```
score = 1.0 / (1.0 + distance)
```

This maps a `[0, ∞)` distance into a `[0, 1]` score where 1.0 means
identical and 0.5 means moderate similarity. The conversion is
monotonic, so the ranking is preserved — the score is purely cosmetic.

## Performance envelope

Where each layer's latency lives at different corpus sizes:

| Chunks | Embed query (cold) | Embed query (cached) | kNN search | Round-trip total       |
|--------|--------------------|----------------------|------------|------------------------|
| 1 K    | ~50 ms             | <1 ms                | <1 ms      | ~50 ms / ~1 ms         |
| 10 K   | ~50 ms             | <1 ms                | ~5–15 ms   | ~70 ms / ~15 ms        |
| 50 K   | ~50 ms             | <1 ms                | ~30–80 ms  | ~130 ms / ~80 ms       |
| 100 K  | ~50 ms             | <1 ms                | ~50–150 ms | ~200 ms / ~150 ms      |
| 500 K  | ~50 ms             | <1 ms                | ~250–700 ms | ~750 ms / ~700 ms     |

Two things to notice:

1. **Below ~50 K chunks, the embedding step dominates.** The
   query-side embed cache is what brings repeated queries down to
   ~1 ms total — second time you ask a question, you skip llama.cpp
   entirely.
2. **Above ~100 K chunks, the linear-scan kNN dominates.** The
   embedder is already as fast as it's going to be; you'd start
   feeling the absence of an ANN index here.

For most cspace-search use cases (one project's code + commits +
context) you'll live in the top half of that table where everything's
snappy.
