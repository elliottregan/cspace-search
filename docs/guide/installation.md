# Installation

cspace-search ships as a single statically-linked binary with no
runtime dependencies beyond a working OS.

## Homebrew

```sh
brew install elliottregan/cspace/cspace-search
```

::: warning Tap not yet published
The Homebrew tap will go up alongside the first tagged release. Until
then, use [Build from source](#build-from-source) below.
:::

## Build from source

You'll need a Rust toolchain (1.74+) and `cmake` (the embedding
backend, `llama-cpp-2`, vendors and builds llama.cpp's C++ source).

```sh
git clone https://github.com/elliottregan/cspace-search.git
cd cspace-search
cargo build --release
sudo cp target/release/cspace-search /usr/local/bin/
```

A clean build takes ~5 minutes the first time (most of which is
llama.cpp compiling). Incremental rebuilds are sub-second.

## Verify

```sh
cspace-search --version
cspace-search status --root .
```

## System requirements

- **macOS** (any version supported by current Rust) or **Linux**
  (x86_64 or aarch64).
- **~150 MB** of disk for the binary.
- **~80 MB** more for the embedding model on first `init` (downloaded
  to `~/.cache/huggingface/hub/`).
- **Apple Silicon recommended** for fast embedding via Metal
  acceleration. CPU-only systems work but are 5–20× slower at index
  time. Search-time latency is not meaningfully affected by hardware
  once the model is loaded.

## Uninstall

```sh
# Binary
sudo rm /usr/local/bin/cspace-search

# Indices (per-machine, all projects)
rm -rf ~/.cspace-search

# Model file (per-machine, shared with anything else using the
# Hugging Face cache)
rm -rf ~/.cache/huggingface/hub/models--jinaai--jina-embeddings-v5-text-nano-retrieval
```
