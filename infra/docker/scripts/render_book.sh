#!/usr/bin/env bash
set -euo pipefail

BOOK_SOURCE="${BOOK_SOURCE:-/book}"
WIDGET_DIR="${WIDGET_DIR:-/widget}"
OUTPUT_DIR="${OUTPUT_DIR:-/output}"
WORK_DIR="${WORK_DIR:-/work/book_src}"

echo "=== EarthRISE Book Renderer ==="
echo "Book source: ${BOOK_SOURCE}"
echo "Widget dir:  ${WIDGET_DIR}"
echo "Output dir:  ${OUTPUT_DIR}"

# 1. Prepare directories
mkdir -p "$(dirname "$WORK_DIR")" "$OUTPUT_DIR"

# 2. Copy book source to working directory (remove broken .git pointer from submodule copy)
echo "--- Copying book source ---"
rm -rf "$WORK_DIR"
cp -r "$BOOK_SOURCE" "$WORK_DIR"
rm -rf "$WORK_DIR/.git"

# 3. Inject Quarto profile overlay
echo "--- Injecting chat profile overlay ---"
cp "$WIDGET_DIR/_quarto-chat.yml" "$WORK_DIR/_quarto-chat.yml"

# 4. Inject chat widget HTML
echo "--- Injecting chat widget ---"
mkdir -p "$WORK_DIR/_includes"
cp "$WIDGET_DIR/chat.html" "$WORK_DIR/_includes/chat.html"

# 5. Render
echo "--- Rendering book ---"
cd "$WORK_DIR"
quarto render --profile chat

# 6. Publish to output volume via staging
echo "--- Publishing output ---"
STAGING=$(mktemp -d /tmp/book_html.XXXXXX)
cp -a "$WORK_DIR/_book"/. "$STAGING"/
rm -rf "${OUTPUT_DIR:?}"/*
cp -a "$STAGING"/. "$OUTPUT_DIR"/
rm -rf "$STAGING"

echo "=== Render complete ==="
ls -la "$OUTPUT_DIR/"
