# Poster

LaTeX poster using `baposter` (A0 portrait, 841×1189mm).

## Build

```bash
cd poster && pdflatex poster.tex
```

`baposter.cls` is checked in locally (not installed system-wide).

## Layout

6-column grid. Each box is placed with `column=`, `span=`, `below=`:

```
┌──────────┬───────────────────────────────┐
│          │  Methodology (span=4)          │
│  Left    │  Excalidraw diagram            │
│  col     ├───────────────┬───────────────┤
│  (span=2)│  Results      │  Observations │
│          │  (span=2)     │  + Next Steps │
└──────────┴───────────────┴───────────────┘
```

To resize: change `span=` values (must sum to 6 per row).

## Diagrams (Excalidraw → LaTeX)

Diagrams live in Obsidian as `.excalidraw` files, get exported and embedded as images.

**Workflow:**

1. Create/edit the diagram in Obsidian's Excalidraw plugin (`~/vault/Excalidraw/`)
2. Export as SVG: in Excalidraw, use the export menu → SVG (or PNG for simpler embedding)
3. Copy the exported file to `poster/graphics/`
4. Embed in `poster.tex`:
   ```latex
   \includegraphics[width=0.90\linewidth]{graphics/methodology.png}
   ```

**Render via CLI** (alternative to manual export):
```bash
cd ~/.claude/skills/obsidian/excalidraw-references
uv run python render_excalidraw.py ~/vault/Excalidraw/<file>.excalidraw --output poster/graphics/<name>.png
```

First-time setup for the renderer:
```bash
cd ~/.claude/skills/obsidian/excalidraw-references && uv sync && uv run playwright install chromium
```

## Content Editing

Body text inside `\headerbox{Title}{options}{...}` is plain LaTeX — just type.

Helper macros for styled elements:

| Macro | What it does |
|-------|-------------|
| `\callout{text}` | Coral-bordered callout box |
| `\bubble{color}{LABEL}{text}` | Chat bubble (system/user/response) |
| `\concept{bg}{title}{description}` | Concept card with colored background |
| `\barrow{label}{color}{0.8}{80\%}` | Horizontal bar chart row |
| `\hlcoral{text}`, `\hlsage{text}`, `\hlnavy{text}` | Bold colored highlight |
| `\muted{text}` | Gray muted text |

## Colors

| Name | Hex | Use |
|------|-----|-----|
| `navy` | #0B1D33 | Box headers, primary text |
| `navymid` | #15304F | RQ band background |
| `coral` | #E8453C | Accent, callouts |
| `sage` | #2D7A6B | Methodology, results highlights |
| `warmgray` | #6B6560 | Muted text, labels |
