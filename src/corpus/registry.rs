//! Pluggable corpus-builder registry.
//!
//! Decouples the corpus factory from the set of known types: instead
//! of a hardcoded `match cfg.type { ... }` in `config::runtime`, each
//! type registers a builder callback under its type-name string.
//! `with_builtins()` wires the in-tree types (`files`, `commits`); a
//! future release can accept external builders (e.g. a `slack` corpus
//! plugin) without editing runtime.rs.
//!
//! Why a registry: the old dispatch was a closed enum hiding behind a
//! match. Adding a new corpus meant editing `runtime.rs`, which is
//! fine when we own every corpus but fights us the moment we want
//! downstream consumers to extend the tool. Open/closed: the registry
//! is open to extension by construction, closed (to us) for the
//! in-tree defaults.

use super::Corpus;
use crate::config::CorpusConfig;
use std::collections::HashMap;

/// Function that turns a resolved `CorpusConfig` + its id into a
/// boxed `Corpus` trait object, or reports why it couldn't.
pub type Builder =
    Box<dyn Fn(&str, &CorpusConfig) -> anyhow::Result<Box<dyn Corpus>> + Send + Sync>;

#[derive(Default)]
pub struct CorpusRegistry {
    builders: HashMap<String, Builder>,
}

impl CorpusRegistry {
    /// Empty registry with no builders. Rarely useful directly —
    /// callers almost always want [`Self::with_builtins`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Registry seeded with every in-tree corpus type.
    pub fn with_builtins() -> Self {
        let mut r = Self::new();
        r.register("files", |id, cfg| {
            let fc = super::FileCorpus::from_config(id, cfg)?;
            Ok(Box::new(fc) as Box<dyn Corpus>)
        });
        r.register("commits", |_id, cfg| {
            let cc = super::CommitCorpus {
                limit: cfg.limit.max(0) as usize,
            };
            Ok(Box::new(cc) as Box<dyn Corpus>)
        });
        r
    }

    /// Register (or replace) a builder for `type_name`. The closure
    /// is invoked once per `build()` call with the resolved
    /// `CorpusConfig`; it runs on the hot path of every `init` and
    /// `search` invocation, so keep it cheap.
    pub fn register<F>(&mut self, type_name: &str, f: F) -> &mut Self
    where
        F: Fn(&str, &CorpusConfig) -> anyhow::Result<Box<dyn Corpus>> + Send + Sync + 'static,
    {
        self.builders.insert(type_name.to_string(), Box::new(f));
        self
    }

    /// Build a corpus by type name. Returns `None` if no builder is
    /// registered for that name — caller is responsible for turning
    /// that into the appropriate user-facing error.
    pub fn build(
        &self,
        type_name: &str,
        id: &str,
        cfg: &CorpusConfig,
    ) -> Option<anyhow::Result<Box<dyn Corpus>>> {
        self.builders.get(type_name).map(|f| f(id, cfg))
    }

    /// Sorted list of registered type names. Used for error messages
    /// that want to say "known types: …".
    pub fn known_types(&self) -> Vec<String> {
        let mut v: Vec<String> = self.builders.keys().cloned().collect();
        v.sort();
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::Record;

    #[test]
    fn with_builtins_registers_files_and_commits() {
        let r = CorpusRegistry::with_builtins();
        let known = r.known_types();
        assert!(known.contains(&"files".to_string()));
        assert!(known.contains(&"commits".to_string()));
    }

    #[test]
    fn build_missing_type_returns_none() {
        let r = CorpusRegistry::with_builtins();
        assert!(r
            .build("bogus", "bogus", &CorpusConfig::default())
            .is_none());
    }

    #[test]
    fn external_register_takes_precedence() {
        #[derive(Debug)]
        struct Stub;
        impl Corpus for Stub {
            fn id(&self) -> &'static str {
                "stub"
            }
            fn collection(&self, _: &std::path::Path) -> String {
                "stub-00000000".into()
            }
            fn enumerate(&self, _: &std::path::Path) -> anyhow::Result<Vec<Record>> {
                Ok(Vec::new())
            }
        }
        let mut r = CorpusRegistry::with_builtins();
        r.register("stub", |_, _| Ok(Box::new(Stub) as Box<dyn Corpus>));

        let built = r
            .build("stub", "anything", &CorpusConfig::default())
            .expect("builder present")
            .unwrap();
        assert_eq!(built.id(), "stub");
    }
}
