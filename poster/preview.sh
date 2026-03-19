#!/bin/bash
# Render poster HTML → PDF at A0 size and open it
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="$DIR/poster-preview.pdf"

/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
  --headless --disable-gpu \
  --print-to-pdf="$OUT" \
  --no-margins \
  --paper-width=33.11 --paper-height=46.81 \
  "file://$DIR/poster.html" 2>/dev/null

open "$OUT"
echo "✅ $OUT"
