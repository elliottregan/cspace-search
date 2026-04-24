# Porting status — Go → Rust

Source: `github.com/elliottregan/cspace/search/*` (Go) and `cmd/cspace-search{,-mcp}`.

Rewrite is phase-based so each phase is small and verifiable on its own.

## Phase 1 — Skeleton + CI ✅
- [x] `Cargo.toml`, directory tree
- [x] `clap`-based CLI with subcommand stubs (init, search, status, clusters, mcp)
- [x] CI: `cargo check` + `cargo test` on push
- [ ] `cargo-dist` release pipeline (Homebrew tap, GitHub release artifacts)

## Phase 2 — Config + corpus walkers
- [ ] Port `search/config/` → `src/config/`. YAML schema stays byte-identical so
      users can copy their `.cspace/search.yaml` over without edits.
- [ ] Port `search/corpus/commits.go` → `src/corpus/commits.rs` using `git2`.
- [ ] Port `search/corpus/code.go` → `src/corpus/code.rs` using `ignore`.
- [ ] Port `search/corpus/context.go` → `src/corpus/context.rs` (local FS walk).
- [ ] Port `search/corpus/issues.go` → `src/corpus/issues.rs` shelling out to `gh`.

## Phase 3 — Storage (sqlite-vec)
- [ ] Replace `search/qdrant/` with `src/index/sqlite_vec.rs`.
- [ ] Preserve the collection-per-corpus model from Go so MCP tool schemas don't
      change.
- [ ] Content-hash skip cache (`search/index/hashes.go`) → `src/index/hashes.rs`.
- [ ] Decide: one DB file per project (keyed by project path hash) OR one DB
      per corpus. The Go version used one qdrant collection per corpus inside a
      per-project qdrant instance — equivalent would be one sqlite file per
      project with multiple `vec_*` tables.

## Phase 4 — Embedding (Candle)
- [ ] Replace `search/embed/` HTTP client with `src/embed/candle.rs`.
- [ ] Load Jina v5 nano (GGUF) — must match the model cspace ships today
      (`jinaai/jina-embeddings-v5-text-nano-retrieval-GGUF`).
- [ ] Preserve the "Document: " / "Query: " prefix convention used by Jina v5
      for retrieval tasks.
- [ ] First-run model download to `~/.cspace-search/models/` with progress.
- [ ] Verify cosine similarity output matches the current llama-server pipeline
      on the existing test corpus within tolerance.

## Phase 5 — Query + ranking
- [ ] Port `search/query/` → `src/query/`.
- [ ] Preserve the multi-corpus fanout + rank-fusion behaviour.

## Phase 6 — MCP server
- [ ] Port `cmd/cspace-search-mcp/` + `search/mcp/` → `src/mcp/` using `rmcp`.
- [ ] Tool names and JSON schemas byte-identical to the Go version so existing
      MCP clients and cspace's in-container registration continue to work.

## Phase 7 — Clusters (deferred, optional)
- [ ] Keep the Docker-based `reduce-api` (mindthemath/reduce-api) + `hdbscan-api`.
      No production Rust port of PaCMAP exists; HDBSCAN has one in
      `linfa-clustering` but PaCMAP doesn't.
- [ ] `src/commands/clusters.rs` reads `--reduce-url` / `--hdbscan-url` with
      sane defaults (`localhost:8000` / `localhost:8090`).
- [ ] Bundle a `docker-compose.yml` in the repo root so `docker compose up -d`
      stands the services up. No cspace dependency.
- [ ] If the services aren't reachable, fail with a clear "run `docker compose
      up -d` in this repo" message instead of hanging.

## Phase 8 — Release
- [ ] `cargo-dist` Homebrew tap + GitHub release
- [ ] Model file distribution: host GGUF on HuggingFace, download on first run
- [ ] macOS code signing / notarization (inherit from cspace's setup if reusable)

## Phase 9 — cspace cleanup (separate repo)
Done in cspace, not here. Tracked separately once this repo can replace the
in-cspace `cspace search` binary.
- [ ] Remove `cmd/cspace-search/`, `cmd/cspace-search-mcp/`, `search/` from
      cspace.
- [ ] Remove `lib/templates/docker-compose.project.yml` search stack (qdrant,
      llama-server, llama-clustering).
- [ ] Remove `lib/hdbscan-api/` from cspace (moved to cspace-search if kept).
- [ ] Remove `cspace search` subcommand wiring + `lib/scripts/init-claude-plugins.sh`
      registration of `cspace-search` MCP.
- [ ] Remove `lib/templates/docker-compose.search.yml` (the per-instance MCP
      sidecar).
- [ ] Drop search-related tests (CommitsStaleness, status suite, etc.).
- [ ] Traefik labels for search.* hostnames go away.
- [ ] Ship as cspace `v1.0.0`.
