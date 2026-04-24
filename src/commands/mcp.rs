//! `cspace-search mcp` — stdio MCP server.
//!
//! Exposes two tools:
//!   - `search` — kNN over one of the enabled corpora.
//!   - `search_status` — reports the master-switch state and per-corpus
//!     runnability, matching the `status --json` shape.
//!
//! The server loads the embedder eagerly on startup (first-use cost is
//! a ~80MB GGUF download from HuggingFace) so the first `search` call
//! doesn't surprise a client with a minutes-long latency spike. The
//! sqlite database is opened once and reused across calls — WAL mode
//! keeps concurrent `cspace-search init` from blocking reads.
//!
//! Concession (see
//! `.cspace/context/decisions/2026-04-24-mcp-single-search-tool.md`):
//! the original plan called for one MCP tool per enabled corpus
//! (`search_code`, `search_commits`, …). rmcp 1.5's tool registration
//! is macro-driven at compile time; synthesizing N tools at runtime
//! requires hand-rolling `ServerHandler`. v0.1 ships a single `search`
//! tool with a required `corpus` parameter instead — functionally
//! equivalent for any MCP client, less code to maintain. Revisit if
//! MCP clients ever start surfacing tools in a way where
//! tool-name-per-corpus would improve discoverability.

use crate::config;
use crate::embed::cache::{CachedEmbedder, EmbedCache};
use crate::embed::llama::LlamaEmbedder;
use crate::embed::{Embedder, FakeEmbedder};
use crate::index::sqlite::SqliteUpserter;
use crate::index::Upserter;
use crate::query::{self, Hit};
use crate::util;
use clap::Parser;
use rmcp::{
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{CallToolResult, Content, ServerCapabilities, ServerInfo},
    service::ServiceExt,
    tool, tool_handler, tool_router, ErrorData, ServerHandler,
};
use schemars::JsonSchema;
use serde::Deserialize;
use std::path::PathBuf;
use std::sync::Arc;

/// Run the stdio MCP server exposing `search` and `search_status`.
#[derive(Parser, Debug)]
pub struct Args {
    /// Project root. Defaults to the nearest ancestor directory with a `.git/`.
    #[arg(long)]
    pub root: Option<PathBuf>,

    /// Use the deterministic FakeEmbedder instead of loading Jina v5.
    /// Matches the flag on `init` and `search`; a caller indexing with
    /// `--fake-embedder` must query with `--fake-embedder` too (the
    /// fingerprint check ensures that mismatches fail loudly rather
    /// than silently return garbage).
    #[arg(long)]
    pub fake_embedder: bool,

    /// Embedding dim for `--fake-embedder`. Ignored otherwise.
    #[arg(long, default_value_t = crate::embed::llama::DEFAULT_DIM)]
    pub dim: usize,

    /// Skip the global embedding cache. See `init --help` for the
    /// tradeoff.
    #[arg(long)]
    pub no_embed_cache: bool,
}

pub fn run(args: Args) -> anyhow::Result<()> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    rt.block_on(run_async(args))
}

async fn run_async(args: Args) -> anyhow::Result<()> {
    let root = args.root.map(Ok).unwrap_or_else(util::find_project_root)?;

    // Eagerly resolve deps so we fail fast with a clear error before
    // handing stdio over to the MCP transport.
    let embedder: Arc<dyn Embedder> = {
        let inner: Box<dyn Embedder> = if args.fake_embedder {
            Box::new(FakeEmbedder::new(args.dim))
        } else {
            // First-use cost of the HF download goes to stderr; the MCP
            // transport owns stdout.
            eprintln!("cspace-search mcp: loading Jina v5 nano retrieval (first use downloads ~80MB)...");
            Box::new(LlamaEmbedder::jina_v5_nano_retrieval()?)
        };
        if args.no_embed_cache {
            Arc::<Box<dyn Embedder>>::from(inner)
        } else {
            let cache = EmbedCache::open(util::embed_cache_path()?)?;
            Arc::new(CachedEmbedder::new(inner, cache))
        }
    };
    let db_path = util::index_db_path(&root)?;
    let store = Arc::new(SqliteUpserter::open(&db_path)?);

    let server = SearchServer::new(root, embedder, store);

    // `stdio` consumes the process stdin/stdout pair; log output must
    // therefore go to stderr only.
    let transport = (tokio::io::stdin(), tokio::io::stdout());
    let service = server.serve(transport).await?;
    service.waiting().await?;
    Ok(())
}

/// Arguments for the `search` tool. A required `corpus` selects which
/// collection to query; the rest are shaping knobs.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchArgs {
    /// Which indexed corpus to search (e.g. "code", "commits", "context").
    pub corpus: String,
    /// Natural-language query.
    pub query: String,
    /// Max hits to return. Clamped to `[1, 50]`; defaults to 10.
    #[serde(default)]
    pub limit: Option<usize>,
    /// Glob applied to each hit's `path`. Hits that don't match are
    /// dropped post-search. Standard shell globbing (`*`, `**`, `?`).
    #[serde(default)]
    pub path_filter: Option<String>,
    /// Comma-separated list of `kind` values (e.g. "commit", "finding").
    /// Hits whose `kind` isn't in the list are dropped post-search.
    #[serde(default)]
    pub kind_filter: Option<String>,
    /// When true, read the file and slice `line_start..=line_end` into
    /// each hit's `preview` field. Costs one file read per hit.
    #[serde(default)]
    pub include_preview: bool,
}

/// Arguments for `search_status`. No inputs.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct StatusArgs {}

/// MCP server state. Built once in `run_async`; every tool call shares
/// this single embedder + store pair.
#[derive(Clone)]
pub struct SearchServer {
    project_root: PathBuf,
    embedder: Arc<dyn Embedder>,
    store: Arc<SqliteUpserter>,
    // Populated by the `#[tool_router]` macro and consumed by
    // `#[tool_handler]`'s generated impl — neither path the dead-code
    // analyzer tracks directly.
    #[allow(dead_code)]
    tool_router: ToolRouter<Self>,
}

impl SearchServer {
    pub fn new(
        project_root: PathBuf,
        embedder: Arc<dyn Embedder>,
        store: Arc<SqliteUpserter>,
    ) -> Self {
        Self {
            project_root,
            embedder,
            store,
            tool_router: Self::tool_router(),
        }
    }
}

#[tool_router]
impl SearchServer {
    #[tool(
        description = "Semantic search across an indexed corpus. Returns ranked hits with path, line range, kind, score, and corpus-specific extras."
    )]
    async fn search(
        &self,
        Parameters(args): Parameters<SearchArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let hits = self.do_search(args).map_err(|e| {
            ErrorData::internal_error(format!("search failed: {e}"), None)
        })?;
        let json = serde_json::to_string(&hits).map_err(|e| {
            ErrorData::internal_error(format!("serialize hits: {e}"), None)
        })?;
        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    #[tool(
        description = "Index status: per-corpus runnability, dim, fingerprint, and whether vectors are present. Does not re-index."
    )]
    async fn search_status(
        &self,
        Parameters(_): Parameters<StatusArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let status = self.do_status().map_err(|e| {
            ErrorData::internal_error(format!("status failed: {e}"), None)
        })?;
        let json = serde_json::to_string(&status).map_err(|e| {
            ErrorData::internal_error(format!("serialize status: {e}"), None)
        })?;
        Ok(CallToolResult::success(vec![Content::text(json)]))
    }
}

#[tool_handler]
impl ServerHandler for SearchServer {
    fn get_info(&self) -> ServerInfo {
        // ServerInfo is #[non_exhaustive]; mutate a default rather
        // than using struct-literal syntax.
        let mut info = ServerInfo::default();
        info.capabilities = ServerCapabilities::builder().enable_tools().build();
        info.instructions = Some(
            "cspace-search: local-first semantic search over code, commits, and context. \
             Call `search` with {corpus, query} to query an indexed corpus; \
             call `search_status` for the per-corpus runnability report."
                .into(),
        );
        info
    }
}

impl SearchServer {
    fn do_search(&self, args: SearchArgs) -> anyhow::Result<Vec<Hit>> {
        let cfg = config::load(&self.project_root)?;
        let runtime =
            config::runtime::build_with_config(&self.project_root, &args.corpus, cfg)?;

        // Over-fetch so post-search filters (path/kind) don't starve
        // the caller of results when a filter is selective. We ask the
        // query layer for its default cap; its internal clamp handles
        // the upper bound.
        let requested_limit = args.limit.unwrap_or(query::DEFAULT_TOP_K);
        let fetch_top_k = if args.path_filter.is_some() || args.kind_filter.is_some() {
            requested_limit.saturating_mul(4).min(query::MAX_TOP_K)
        } else {
            requested_limit
        };

        let env = query::run(query::RunConfig {
            corpus: runtime.corpus.as_ref(),
            embedder: self.embedder.as_ref(),
            searcher: self.store.as_ref(),
            project_root: &self.project_root,
            query: &args.query,
            top_k: fetch_top_k,
            with_cluster: false,
        })?;

        let kind_allow: Option<Vec<String>> = args.kind_filter.as_ref().map(|s| {
            s.split(',')
                .map(|k| k.trim().to_string())
                .filter(|k| !k.is_empty())
                .collect()
        });
        let path_pat = match args.path_filter.as_ref() {
            Some(pat) => Some(
                glob::Pattern::new(pat)
                    .with_context_glob(pat)?,
            ),
            None => None,
        };

        let mut out: Vec<Hit> = env
            .results
            .into_iter()
            .filter(|h| match &kind_allow {
                Some(allow) if !allow.is_empty() => allow.iter().any(|k| k == &h.kind),
                _ => true,
            })
            .filter(|h| match &path_pat {
                Some(p) => p.matches(&h.path),
                None => true,
            })
            .collect();

        if args.include_preview {
            for h in &mut out {
                h.preview = build_preview(&self.project_root, h);
            }
        }
        out.truncate(requested_limit.clamp(1, query::MAX_TOP_K));
        Ok(out)
    }

    fn do_status(&self) -> anyhow::Result<serde_json::Value> {
        let cfg = config::load(&self.project_root)?;
        let embedder_fp = self.embedder.fingerprint();
        let embedder_dim = self.embedder.dim();

        let mut corpora = serde_json::Map::new();
        for (id, cc) in &cfg.corpora {
            let runnable = cfg.enabled && cc.enabled;
            let collection = if runnable {
                match config::runtime::build_with_config(
                    &self.project_root,
                    id,
                    cfg.clone(),
                ) {
                    Ok(rt) => Some(rt.corpus.collection(&self.project_root)),
                    Err(_) => None,
                }
            } else {
                None
            };
            // `indexed` == there's at least one row in the collection.
            // A missing collection reads as zero rows (the store maps
            // "no such table" to empty).
            let indexed = match &collection {
                Some(name) => self
                    .store
                    .as_ref()
                    .existing_points(name)
                    .map(|m| !m.is_empty())
                    .unwrap_or(false),
                None => false,
            };
            corpora.insert(
                id.clone(),
                serde_json::json!({
                    "enabled": cc.enabled,
                    "runnable": runnable,
                    "indexed": indexed,
                    "collection": collection,
                }),
            );
        }

        Ok(serde_json::json!({
            "root": self.project_root.display().to_string(),
            "enabled": cfg.enabled,
            "embedder": {
                "dim": embedder_dim,
                "fingerprint": embedder_fp,
            },
            "corpora": corpora,
            // Staleness detection is not implemented in v0.1: we don't
            // re-enumerate corpora at status time (would defeat the
            // "fast, read-only" nature of the tool). Consumers that
            // want stale signal should run `init` and compare the
            // `embedded`/`orphans_deleted` counts.
            "stale": serde_json::Value::Null,
        }))
    }
}

/// Slice `line_start..=line_end` out of the file at `project_root/path`
/// and return it. Returns an empty string on any failure — preview is
/// best-effort, a bad read should never fail the whole tool call.
fn build_preview(project_root: &std::path::Path, h: &Hit) -> String {
    if h.path.is_empty() || h.line_start == 0 {
        return String::new();
    }
    let path = project_root.join(&h.path);
    let body = match std::fs::read_to_string(&path) {
        Ok(b) => b,
        Err(_) => return String::new(),
    };
    let lines: Vec<&str> = body.lines().collect();
    let start = (h.line_start as usize).saturating_sub(1);
    let end = (h.line_end as usize).min(lines.len());
    if start >= lines.len() || start >= end {
        return String::new();
    }
    lines[start..end].join("\n")
}

/// Thin wrapper so glob's `PatternError` gets attached context.
trait PatternErrorExt<T> {
    fn with_context_glob(self, pat: &str) -> anyhow::Result<T>;
}

impl<T> PatternErrorExt<T> for std::result::Result<T, glob::PatternError> {
    fn with_context_glob(self, pat: &str) -> anyhow::Result<T> {
        self.map_err(|e| anyhow::anyhow!("invalid path_filter {pat:?}: {e}"))
    }
}

#[cfg(test)]
mod tests {
    //! These tests exercise `do_search` / `do_status` directly without
    //! a real MCP transport — the tool wrappers are one-line JSON
    //! serializations over these methods, so unit-testing the business
    //! logic is what gives confidence. A round-trip through the stdio
    //! transport would add `tokio::io::duplex` plumbing for no extra
    //! coverage of the search path.

    use super::*;
    use crate::embed::FakeEmbedder;
    use crate::index::{self as idx, sqlite::SqliteUpserter};

    fn seed_project(dir: &std::path::Path) {
        let ctx = dir.join(".cspace").join("context");
        std::fs::create_dir_all(ctx.join("findings")).unwrap();
        std::fs::write(ctx.join("principles.md"), "keep it simple").unwrap();
        std::fs::write(ctx.join("roadmap.md"), "ship in phases").unwrap();
        std::fs::write(ctx.join("findings/a.md"), "alpha finding").unwrap();
        // Enable the context corpus explicitly — the default has it off.
        std::fs::write(
            dir.join("search.yaml"),
            "enabled: true\ncorpora:\n  context:\n    enabled: true\n",
        )
        .unwrap();
    }

    fn build_server(dir: &std::path::Path) -> SearchServer {
        let embedder: Arc<dyn Embedder> = Arc::new(FakeEmbedder::new(16));
        let store = Arc::new(SqliteUpserter::in_memory().unwrap());

        // Index the context corpus so search has something to hit.
        let cfg = config::load(dir).unwrap();
        let runtime =
            config::runtime::build_with_config(dir, "context", cfg.clone()).unwrap();
        idx::run(idx::RunConfig {
            corpus: runtime.corpus.as_ref(),
            embedder: embedder.as_ref(),
            upserter: store.as_ref(),
            project_root: dir,
            batch_size: 8,
            progress: None,
        })
        .unwrap();

        SearchServer::new(dir.to_path_buf(), embedder, store)
    }

    #[test]
    fn do_search_returns_ranked_hits_for_indexed_corpus() {
        let dir = tempfile::tempdir().unwrap();
        seed_project(dir.path());
        let server = build_server(dir.path());

        // FakeEmbedder is content-addressed: the query's vector is
        // byte-identical to the doc's when the text matches.
        let hits = server
            .do_search(SearchArgs {
                corpus: "context".into(),
                query: "Context (principles): .cspace/context/principles.md\n\nkeep it simple"
                    .into(),
                limit: Some(3),
                path_filter: None,
                kind_filter: None,
                include_preview: false,
            })
            .unwrap();

        assert!(!hits.is_empty());
        assert_eq!(hits[0].path, ".cspace/context/principles.md");
    }

    #[test]
    fn do_search_applies_kind_filter() {
        let dir = tempfile::tempdir().unwrap();
        seed_project(dir.path());
        let server = build_server(dir.path());

        let hits = server
            .do_search(SearchArgs {
                corpus: "context".into(),
                // Query an arbitrary phrase; the kind filter is what
                // we're testing — only `finding` records should pass.
                query: "anything".into(),
                limit: Some(10),
                path_filter: None,
                kind_filter: Some("finding".into()),
                include_preview: false,
            })
            .unwrap();

        for h in &hits {
            assert_eq!(h.kind, "finding", "kind_filter failed to drop {h:?}");
        }
    }

    #[test]
    fn do_search_applies_path_filter_glob() {
        let dir = tempfile::tempdir().unwrap();
        seed_project(dir.path());
        let server = build_server(dir.path());

        let hits = server
            .do_search(SearchArgs {
                corpus: "context".into(),
                query: "anything".into(),
                limit: Some(10),
                path_filter: Some(".cspace/context/findings/**".into()),
                kind_filter: None,
                include_preview: false,
            })
            .unwrap();

        for h in &hits {
            assert!(
                h.path.starts_with(".cspace/context/findings/"),
                "path_filter failed to drop {h:?}"
            );
        }
    }

    #[test]
    fn do_status_reports_embedder_fingerprint_and_indexed_flag() {
        let dir = tempfile::tempdir().unwrap();
        seed_project(dir.path());
        let server = build_server(dir.path());

        let status = server.do_status().unwrap();
        assert_eq!(status["enabled"], serde_json::Value::Bool(true));
        assert_eq!(
            status["embedder"]["fingerprint"]
                .as_str()
                .unwrap_or_default(),
            "fake:dim=16"
        );
        assert_eq!(status["embedder"]["dim"], serde_json::json!(16));
        assert_eq!(status["corpora"]["context"]["runnable"], serde_json::Value::Bool(true));
        assert_eq!(
            status["corpora"]["context"]["indexed"],
            serde_json::Value::Bool(true)
        );
        // Deferred per docs; must be explicitly null, not absent.
        assert!(status["stale"].is_null());
    }

    #[test]
    fn do_status_reports_disabled_corpus_as_not_runnable() {
        let dir = tempfile::tempdir().unwrap();
        seed_project(dir.path());
        // Override commits to disabled for this test.
        std::fs::write(
            dir.path().join("search.yaml"),
            "enabled: true\n\
             corpora:\n  \
               context:\n    enabled: true\n  \
               commits:\n    enabled: false\n",
        )
        .unwrap();
        let server = build_server(dir.path());

        let status = server.do_status().unwrap();
        assert_eq!(
            status["corpora"]["commits"]["runnable"],
            serde_json::Value::Bool(false)
        );
    }
}
