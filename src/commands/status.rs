use crate::config;
use clap::Parser;
use std::path::PathBuf;

/// Print index staleness + size + model version.
///
/// Partial port: currently reports only the master-switch state from
/// `search.yaml`. Real staleness + size come with Phase 3 (sqlite-vec
/// storage) and Phase 4 (embedding model provenance).
#[derive(Parser, Debug)]
pub struct Args {
    /// Project root. Defaults to the current working directory.
    #[arg(long)]
    pub root: Option<PathBuf>,

    /// Restrict output to a single corpus (code, commits, context, issues).
    /// Reports whether that corpus would run, applying the master-switch
    /// and per-corpus gating that every search path respects.
    #[arg(long)]
    pub corpus: Option<String>,

    /// Output machine-readable JSON.
    #[arg(long)]
    pub json: bool,
}

pub fn run(args: Args) -> anyhow::Result<()> {
    let root = args.root.map(Ok).unwrap_or_else(std::env::current_dir)?;
    let cfg = config::load(&root)?;

    if let Some(ref id) = args.corpus {
        return match config::resolve_corpus(&cfg, id, &root) {
            Ok(cc) => {
                if args.json {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&serde_json::json!({
                            "corpus": id,
                            "runnable": true,
                            "config": { "enabled": cc.enabled, "max_bytes": cc.max_bytes, "limit": cc.limit },
                        }))?
                    );
                } else {
                    println!("{id}: runnable");
                }
                Ok(())
            }
            Err(e) => {
                if args.json {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&serde_json::json!({
                            "corpus": id,
                            "runnable": false,
                            "error": e.to_string(),
                        }))?
                    );
                    Ok(())
                } else {
                    Err(anyhow::Error::from(e))
                }
            }
        };
    }

    if args.json {
        let state = serde_json::json!({
            "root": root.display().to_string(),
            "enabled": cfg.enabled,
            "corpora": cfg.corpora.iter().map(|(id, cc)| {
                (id.clone(), serde_json::json!({
                    "enabled": cc.enabled,
                    "max_bytes": cc.max_bytes,
                    "limit": cc.limit,
                }))
            }).collect::<std::collections::BTreeMap<_, _>>(),
        });
        println!("{}", serde_json::to_string_pretty(&state)?);
        return Ok(());
    }

    println!("root:    {}", root.display());
    println!("enabled: {}", cfg.enabled);
    println!("corpora:");
    for (id, cc) in &cfg.corpora {
        println!(
            "  {id:<8} enabled={} limit={} max_bytes={}",
            cc.enabled, cc.limit, cc.max_bytes
        );
    }
    if !cfg.enabled {
        eprintln!();
        eprintln!("  search is not configured for this project.");
        eprintln!(
            "  set `enabled: true` in {}/search.yaml to activate.",
            root.display()
        );
    }
    Ok(())
}
