# Poster

LaTeX poster using `baposter` (A0 portrait). `baposter.cls` is checked in locally (source: `mloesch/baposter` on GitHub).

## Build

```bash
cd poster && pdflatex poster.tex
```

## Excalidraw Diagrams

Diagrams are `.excalidraw` files (JSON format) in `~/vault/Excalidraw/`. To create or edit them, use the **obsidian skill** — it has an Excalidraw sub-skill (`excalidraw.md`) with full instructions for generating the JSON, color palette, element templates, and a render-view-fix loop.

**Workflow:**

1. Invoke the obsidian skill, read `excalidraw.md`
2. Generate/edit `.excalidraw` JSON in `~/vault/Excalidraw/`
3. Render to PNG and copy to `poster/graphics/`:
   ```bash
   cd ~/.claude/skills/obsidian/excalidraw-references
   uv run python render_excalidraw.py ~/vault/Excalidraw/<file>.excalidraw \
     --output <absolute-path-to>/poster/graphics/<name>.png
   ```
4. Embed in `poster.tex`: `\includegraphics[width=0.90\linewidth]{graphics/<name>.png}`

The user can also edit `.excalidraw` files visually in Obsidian's Excalidraw plugin and export manually.

**First-time renderer setup:**
```bash
cd ~/.claude/skills/obsidian/excalidraw-references && uv sync && uv run playwright install chromium
```
