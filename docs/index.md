---
layout: home

hero:
  name: cspace-search
  text: Local-first semantic search
  tagline: For commits, code, and project context. Single binary, no Docker, no daemon.
  actions:
    - theme: brand
      text: Quick start
      link: /guide/quickstart
    - theme: alt
      text: How it works
      link: /architecture/what-happens-during-a-search

features:
  - title: One binary, zero infrastructure
    details: |
      Ships as a single statically-linked executable. No Docker sidecar,
      no separate vector database, no Python runtime. The model
      downloads on first use; everything else is sqlite.
  - title: Built for agents
    details: |
      The `mcp` subcommand exposes search and status as a stdio MCP
      server. Tool schemas advertise per-corpus kind vocabularies so
      agents (and clients) discover what's available without guessing.
  - title: Incremental by default
    details: |
      Per-chunk content hashes skip re-embedding for unchanged text;
      a global embedding cache catches text the model has seen before.
      Re-indexing after a model swap is essentially free.
---
