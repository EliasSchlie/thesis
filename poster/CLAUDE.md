# Poster

LaTeX poster using `baposter` (A0 portrait). `baposter.cls` is checked in locally (source: `mloesch/baposter` on GitHub).

## Build

```bash
cd poster && pdflatex poster.tex
```

## Embedding Excalidraw Diagrams

1. Create/edit in Obsidian's Excalidraw plugin (`~/vault/Excalidraw/`)
2. Export to `poster/graphics/` (SVG preferred for print sharpness, PNG also works)
3. Embed: `\includegraphics[width=0.90\linewidth]{graphics/<name>.png}`

**CLI render** (alternative to manual export):
```bash
cd ~/.claude/skills/obsidian/excalidraw-references
uv run python render_excalidraw.py ~/vault/Excalidraw/<file>.excalidraw --output poster/graphics/<name>.png
```

First-time renderer setup:
```bash
cd ~/.claude/skills/obsidian/excalidraw-references && uv sync && uv run playwright install chromium
```
