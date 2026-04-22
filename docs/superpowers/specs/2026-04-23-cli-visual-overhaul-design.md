# CLI Visual Overhaul — Agentic Video RAG
**Date:** 2026-04-23  
**Author:** M.S Swaroop  
**Status:** Approved

---

## Goal

Transform the existing `src/cli.py` into a polished, branded terminal experience modelled on the Claude Code CLI aesthetic — custom bot avatar, consistent identity mark, clean conversation layout, and a full-width welcome banner.

---

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Rendering approach | Rich-only upgrade | Zero new dependencies; `rich` and `Pillow` already present |
| Bot avatar | PNG → half-block Unicode pixel art | Renders the actual bot image using Pillow + `▀/▄` + true-color ANSI codes |
| Input prompt symbol | `⚡` | Distinctive, lightweight, Claude Code-style mark |
| Response layout | Inline with bot name header (no box) | Matches Claude Code's conversational flow; existing Panel removed |
| Bot name | "Agentic Video RAG by M.S Swaroop" | Existing brand, extended with author attribution |

---

## Architecture

All changes are confined to `src/cli.py`. No new modules, no new dependencies. `Pillow` is already a project dependency used elsewhere.

### Bot image

The bot PNG (`assets/bot.png`) is bundled with the package. At startup, `_render_bot_image()` reads and resizes it to 28 columns wide using Pillow, then outputs each row of pixel pairs as a `▀` character with true-color ANSI foreground + background codes. Transparent pixels are composited against a dark background (`#121212`).

### Components changed in `src/cli.py`

#### 1. `_render_bot_image(width: int) -> str`
New function. Reads `assets/bot.png`, resizes with Pillow (LANCZOS), and returns a multi-line string of ANSI-colored `▀` characters. Called once at startup. Falls back silently if the image file is missing.

#### 2. `_print_header()`
Replaced. Now prints:
- Bot image pixel art on the left (28 cols wide)
- `Agentic Video RAG` in bold blue to the right
- `by M.S Swaroop · v0.1.0` in dim below the name
- `Ask questions about your lecture. /help for commands.` as tagline
- Vector store path

Layout uses `rich` `Columns` or manual spacing — no Panel box in the main content area. A subtle top/bottom rule (`rich.rule`) frames the header block.

#### 3. `_print_answer()`
Replaced. Instead of a `Panel`, prints:
```
◈ Agentic Video RAG
  <answer text here, word-wrapped>
```
`◈` symbol in bold blue (matching the header identity mark), bot name in bold blue, answer in default color, indented by 2 spaces. No emoji — the actual bot image appears only in the welcome banner.

#### 4. `_print_status()`
Unchanged in behavior. Output remains `[cyan]message[/cyan]` styled lines, prefixed with `  ·` for visual hierarchy.

#### 5. Input prompt in `chat_command()`
`input("\nYou> ")` replaced with `input("\n⚡ ")`.

---

## Data Flow

```
startup
  └─ _render_bot_image()     reads assets/bot.png via Pillow
  └─ _print_header()         prints banner: pixel art + name + tagline

chat loop
  └─ input("⚡ ")            user types question
  └─ _print_status()         "  · Retrieving context..."
  └─ _run_query()            calls AgenticRAG
  └─ _print_status()         "  · Cache: HIT/MISS"
  └─ _print_answer()         "◈ Agentic Video RAG\n  <answer>"
  └─ _print_sources()        (unchanged, shown if /sources on)
```

---

## Error Handling

- If `assets/bot.png` is missing: `_render_bot_image()` returns an empty string; header renders without the image. No crash.
- If Pillow fails to open the image: same fallback, logged to stderr in debug mode only.
- Terminal encoding: stdout is forced to UTF-8 at CLI entry point to support `▀` and `⚡` on Windows.

---

## Files Changed

| File | Change |
|---|---|
| `src/cli.py` | Replace `_print_header`, `_print_answer`; add `_render_bot_image`; update prompt in `chat_command` |
| `assets/bot.png` | New — copy of the blue bot image, bundled with package |
| `pyproject.toml` | Add `package-data` entry to include `assets/bot.png` |

---

## Out of Scope

- `ask`, `ingest`, `build-store`, `eval` commands — visual changes apply only to `chat` mode header and answer display
- `prompt_toolkit` input (deferred to a future upgrade)
- Textual TUI (not needed)
- Streaming/typewriter effect on answers (not in current backend)
