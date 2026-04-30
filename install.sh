#!/usr/bin/env bash
# Install cspace-search — Local-first semantic search for commits, code, and project context.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/elliottregan/cspace-search/main/install.sh | bash
#
# Environment:
#   INSTALL_DIR  — Override install directory (default: /usr/local/bin)
#   VERSION      — Install a specific version (default: latest)
set -euo pipefail

REPO="elliottregan/cspace-search"
INSTALL_DIR="${INSTALL_DIR:-/usr/local/bin}"

# --- Detect OS and architecture ---

OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)
case "$ARCH" in
    x86_64)         ARCH="amd64" ;;
    aarch64|arm64)  ARCH="arm64" ;;
    *)
        echo "Unsupported architecture: $ARCH" >&2
        exit 1
        ;;
esac

case "$OS" in
    darwin|linux) ;;
    *)
        echo "Unsupported OS: $OS" >&2
        exit 1
        ;;
esac

echo "Detected platform: ${OS}/${ARCH}"

# --- Determine version to install ---

if [ -z "${VERSION:-}" ]; then
    echo "Fetching latest release..."
    VERSION=$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" \
        | grep '"tag_name"' \
        | sed -E 's/.*"([^"]+)".*/\1/')

    if [ -z "$VERSION" ]; then
        echo "Error: Could not determine latest version." >&2
        echo "Set VERSION=vX.Y.Z to install a specific version." >&2
        exit 1
    fi
fi

echo "Installing cspace-search ${VERSION}..."

# --- Download binary and checksums ---

ASSET_NAME="cspace-search-${OS}-${ARCH}"
DOWNLOAD_URL="https://github.com/$REPO/releases/download/$VERSION/$ASSET_NAME"
CHECKSUM_URL="https://github.com/$REPO/releases/download/$VERSION/checksums.txt"

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

echo "Downloading ${ASSET_NAME}..."
if ! curl -fsSL -o "$TMPDIR/$ASSET_NAME" "$DOWNLOAD_URL"; then
    echo "Error: Failed to download $DOWNLOAD_URL" >&2
    echo "" >&2
    echo "Available at: https://github.com/$REPO/releases/tag/$VERSION" >&2
    exit 1
fi

# --- Verify checksum ---

if curl -fsSL -o "$TMPDIR/checksums.txt" "$CHECKSUM_URL" 2>/dev/null; then
    echo "Verifying checksum..."
    (
        cd "$TMPDIR"
        if command -v shasum &>/dev/null; then
            grep -F "  ${ASSET_NAME}" checksums.txt | shasum -a 256 --check --quiet
            echo "Checksum verified."
        elif command -v sha256sum &>/dev/null; then
            grep -F "  ${ASSET_NAME}" checksums.txt | sha256sum --check --quiet
            echo "Checksum verified."
        else
            echo "Warning: No checksum tool found, skipping verification." >&2
        fi
    )
else
    echo "Warning: Could not download checksums, skipping verification." >&2
fi

# --- Install binary ---

if [ ! -w "$INSTALL_DIR" ]; then
    echo "Installing to $INSTALL_DIR (requires sudo)..."
    SUDO=sudo
else
    SUDO=
fi

$SUDO mkdir -p "$INSTALL_DIR"
$SUDO cp "$TMPDIR/$ASSET_NAME" "$INSTALL_DIR/cspace-search"
$SUDO chmod +x "$INSTALL_DIR/cspace-search"

# macOS requires binaries to be signed. Cross-compiled binaries from CI have
# no signature, so apply an ad-hoc signature to satisfy Gatekeeper.
if [ "$OS" = "darwin" ] && command -v codesign &>/dev/null; then
    $SUDO codesign -s - "$INSTALL_DIR/cspace-search" 2>/dev/null || true
fi

# --- Verify installation ---

INSTALLED_VERSION=$("$INSTALL_DIR/cspace-search" --version 2>/dev/null || echo "unknown")

echo ""
echo "cspace-search installed successfully!"
echo "  Binary:  $INSTALL_DIR/cspace-search"
echo "  Version: $INSTALLED_VERSION"
echo ""
echo "To get started:"
echo "  1. cd into a project and run: cspace-search init"
echo "  2. Search: cspace-search search 'your query'"
echo "  3. Or expose to an MCP client: cspace-search mcp"
echo ""
echo "Docs: https://github.com/elliottregan/cspace-search"
