# Configuration

cspace-search reads `search.yaml` from the project root. The file has
two layers: a master switch and per-corpus configuration.

## Master switch

```yaml
enabled: true
```

If `enabled` is `false` (the default for fresh projects), no corpus
runs. This is a deliberate guard against an accidental `init` indexing
`node_modules` on a project that hasn't opted in.

## Corpora

```yaml
enabled: true

corpora:
  code:
    enabled: true
    type: files
    source: git-ls-files     # only tracked files
    chunk: { max: 12000, overlap: 200 }

  commits:
    enabled: true
    limit: 1000              # most recent N commits

  context:
    enabled: true
    type: files
    source: filesystem
    record_kind: context
    embed_header: "Context ({subkind}): {path}\n\n"
    path_groups:
      - include:
          - .cspace/context/direction.md
          - .cspace/context/principles.md
          - .cspace/context/roadmap.md
        kind: context
        extra:
          subkind: "{basename_no_ext}"
      - include:
          - .cspace/context/findings/**/*.md
        kind: finding
```

### Per-corpus fields

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `enabled` | bool | `false` | Per-corpus opt-in. Master switch must also be on. |
| `type` | string | `files` for unknown ids, `commits` for `commits` | Selects the corpus implementation. |
| `source` | `git-ls-files` \| `filesystem` \| `walk` | `git-ls-files` | File enumeration strategy. `walk` is `.gitignore`-aware. |
| `record_kind` | string | varies | Default `kind` value for emitted records. |
| `embed_header` | string | `"{kind}: {path}\n\n"` | Templated prefix prepended to each chunk's text before embedding. |
| `chunk` | object | — | `{ max, overlap }` byte sizes. Omit to embed whole files. |
| `max_bytes` | int | varies | Skip files larger than this. |
| `path_groups` | list | — | See below. |
| `excludes` | list | — | Glob patterns to exclude. |
| `limit` | int | varies | Corpus-specific (e.g. commit count). |

### Path groups

`path_groups` lets one corpus emit records under different `kind`
values based on which paths matched. Useful for the `context` corpus,
which classifies `principles.md` as `kind=context` and
`findings/*.md` as `kind=finding`.

Each group:

```yaml
- include: ["glob1", "glob2"]   # paths to match
  kind: my-kind                  # override record_kind for this group
  chunk: { max: 8000 }           # override chunking for this group
  extra:
    arbitrary_key: "{basename_no_ext}"
```

Available template variables in `extra` and `embed_header`:

- `{path}` — relative path
- `{basename}` — file basename
- `{basename_no_ext}` — basename without extension
- `{kind}` — resolved kind for this record

## Where the index goes

Indices land at `~/.cspace-search/<project_hash>.db` where
`<project_hash>` is a sha256 of the absolute project root path. This
namespacing means two clones of the same repo at different paths have
independent indices.

## Defaults

If you omit `search.yaml` entirely, every corpus is **disabled**.
Provide an empty file with `enabled: true` to use the built-in
defaults shipped with cspace-search.

The defaults ship with `code` and `commits` enabled and a sensible
`context` definition that's opt-in. See
[`lib/defaults/search.yaml`](https://github.com/elliottregan/cspace-search/blob/main/lib/defaults/search.yaml)
for the full shipped configuration.
