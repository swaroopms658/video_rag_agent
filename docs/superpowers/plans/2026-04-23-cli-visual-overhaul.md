# CLI Visual Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform `src/cli.py`'s chat mode into a branded terminal experience with a pixel-art bot banner, `⚡` input prompt, and clean `◈`-prefixed inline bot responses.

**Architecture:** All visual changes are confined to `src/cli.py`. The bot PNG is bundled at `src/assets/bot.png` and read at startup by a new `_render_bot_image()` function using Pillow (already a dependency). No new libraries added.

**Tech Stack:** Python 3.10+, `rich`, `Pillow`, `pytest`, `argparse`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/assets/bot.png` | **Create** | Bot image asset bundled with package |
| `src/cli.py` | **Modify** | Add `_render_bot_image`, replace `_print_header`, `_print_answer`, `_print_status`, update prompt symbol, fix Windows UTF-8 |
| `pyproject.toml` | **Modify** | Add `package-data` to include `src/assets/*.png` |
| `tests/__init__.py` | **Create** | Makes `tests/` a package |
| `tests/test_cli_visual.py` | **Create** | Unit tests for all visual functions |
| `_preview_bot.py` | **Delete** | Temporary file, no longer needed |

---

## Task 1: Bundle the bot image and set up test scaffold

**Files:**
- Create: `src/assets/bot.png`
- Modify: `pyproject.toml`
- Create: `tests/__init__.py`
- Create: `tests/test_cli_visual.py` (scaffold only)

- [ ] **Step 1: Copy the bot image into the package**

```bash
mkdir -p src/assets
cp "C:/Users/Sriram/.claude/image-cache/0479f5d7-cda1-4983-9cdb-2428633ec1bb/1.png" src/assets/bot.png
```

Verify: `ls src/assets/` should show `bot.png`.

- [ ] **Step 2: Add package-data to pyproject.toml**

In `pyproject.toml`, the `[tool.setuptools]` section currently reads:

```toml
[tool.setuptools]
packages = ["src"]
```

Replace it with:

```toml
[tool.setuptools]
packages = ["src"]
include-package-data = true

[tool.setuptools.package-data]
src = ["assets/*.png"]
```

- [ ] **Step 3: Create tests package**

Create `tests/__init__.py` as an empty file:

```bash
mkdir -p tests
touch tests/__init__.py
```

- [ ] **Step 4: Create test scaffold**

Create `tests/test_cli_visual.py`:

```python
import sys
import pathlib
import unittest.mock as mock

import pytest


def test_placeholder():
    pass
```

- [ ] **Step 5: Run the scaffold test**

```bash
pytest tests/test_cli_visual.py -v
```

Expected output: `1 passed`

- [ ] **Step 6: Commit**

```bash
git add src/assets/bot.png pyproject.toml tests/__init__.py tests/test_cli_visual.py
git commit -m "feat: bundle bot image asset and add test scaffold"
```

---

## Task 2: Implement `_render_bot_image()`

**Files:**
- Modify: `src/cli.py` (add function, update imports)
- Modify: `tests/test_cli_visual.py`

- [ ] **Step 1: Write failing tests**

Replace the contents of `tests/test_cli_visual.py` with:

```python
import pathlib
import unittest.mock as mock


def test_render_bot_image_returns_string():
    from src.cli import _render_bot_image
    result = _render_bot_image()
    assert isinstance(result, str)


def test_render_bot_image_contains_halfblock_when_image_present():
    from src.cli import _render_bot_image
    result = _render_bot_image()
    if result:
        assert "▀" in result
        assert "\033[" in result


def test_render_bot_image_returns_empty_when_file_missing():
    from src.cli import _render_bot_image
    with mock.patch("pathlib.Path.exists", return_value=False):
        result = _render_bot_image()
    assert result == ""


def test_render_bot_image_returns_empty_on_pillow_error():
    from src.cli import _render_bot_image
    with mock.patch("PIL.Image.open", side_effect=OSError("corrupt")):
        result = _render_bot_image()
    assert result == ""


def test_render_bot_image_custom_width():
    from src.cli import _render_bot_image
    result = _render_bot_image(width=10)
    if result:
        first_line = result.split("\n")[0]
        assert first_line.count("▀") == 10
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_cli_visual.py -v
```

Expected: `ImportError` or `AttributeError` — `_render_bot_image` does not exist yet.

- [ ] **Step 3: Add imports to `src/cli.py`**

At the top of `src/cli.py`, after the existing imports, add:

```python
import pathlib
import sys
from itertools import zip_longest
```

- [ ] **Step 4: Add `_render_bot_image()` to `src/cli.py`**

Add this function after the `APP_NAME` / constant block (around line 18), before `_get_console()`:

```python
def _render_bot_image(width: int = 28) -> str:
    try:
        from PIL import Image
        img_path = pathlib.Path(__file__).parent / "assets" / "bot.png"
        if not img_path.exists():
            return ""
        img = Image.open(img_path).convert("RGBA")
        aspect = img.height / img.width
        height = max(2, int(width * aspect * 0.5) * 2)
        img = img.resize((width, height), Image.LANCZOS)
        pixels = img.load()
        lines = []
        for y in range(0, height, 2):
            line = ""
            for x in range(width):
                r1, g1, b1, a1 = pixels[x, y]
                r2, g2, b2, a2 = pixels[x, y + 1] if y + 1 < height else (18, 18, 18, 255)
                if a1 < 128:
                    r1, g1, b1 = 18, 18, 18
                if a2 < 128:
                    r2, g2, b2 = 18, 18, 18
                line += f"\033[38;2;{r1};{g1};{b1}m\033[48;2;{r2};{g2};{b2}m▀"
            lines.append(line + "\033[0m")
        return "\n".join(lines)
    except Exception:
        return ""
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_cli_visual.py -v
```

Expected: `5 passed`

- [ ] **Step 6: Commit**

```bash
git add src/cli.py tests/test_cli_visual.py
git commit -m "feat: add _render_bot_image with half-block pixel art rendering"
```

---

## Task 3: Replace `_print_header()` with the branded banner

**Files:**
- Modify: `src/cli.py` (replace `_print_header`, update its call site in `chat_command`)
- Modify: `tests/test_cli_visual.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_cli_visual.py`:

```python
def test_print_header_no_crash_with_console(capsys):
    from src.cli import _print_header
    _print_header(None, "data/vector_store.pkl")
    captured = capsys.readouterr()
    assert "Agentic Video RAG" in captured.out


def test_print_header_contains_author(capsys):
    from src.cli import _print_header
    _print_header(None, "data/vector_store.pkl")
    captured = capsys.readouterr()
    assert "M.S Swaroop" in captured.out


def test_print_header_contains_store_path(capsys):
    from src.cli import _print_header
    _print_header(None, "data/my_store.pkl")
    captured = capsys.readouterr()
    assert "my_store.pkl" in captured.out
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_cli_visual.py::test_print_header_no_crash_with_console -v
```

Expected: `FAILED` — current `_print_header` takes `title` and `subtitle`, not `vector_store_path`.

- [ ] **Step 3: Replace `_print_header()` in `src/cli.py`**

Find and replace the entire `_print_header` function (lines ~25–35):

```python
def _print_header(console, vector_store_path: str):
    bot_art = _render_bot_image()
    art_lines = bot_art.split("\n") if bot_art else []

    title_text = [
        "\033[1;34m  Agentic Video RAG\033[0m",
        "\033[2m  by M.S Swaroop · v0.1.0\033[0m",
        "",
        "\033[2m  Ask questions about your lecture.\033[0m",
        "\033[2m  /help for commands.\033[0m",
        f"\033[2m  Store: {vector_store_path}\033[0m",
    ]
    pad_top = max(0, (len(art_lines) - len(title_text)) // 2)
    padded_titles = [""] * pad_top + title_text

    print()
    if console:
        console.rule(style="blue")
    else:
        print("\033[34m" + "─" * 60 + "\033[0m")

    for art, title in zip_longest(art_lines, padded_titles, fillvalue=""):
        sys.stdout.write("  " + art + title + "\n")
    sys.stdout.flush()

    if console:
        console.rule(style="blue")
    else:
        print("\033[34m" + "─" * 60 + "\033[0m")
    print()
```

- [ ] **Step 4: Update the call site in `chat_command()`**

Find this block in `chat_command` (around line 124):

```python
_print_header(
    console,
    "Interactive chat mode",
    f"Using vector store: {vector_store_path}",
)
_print_status(
    console,
    "Commands: /help, /sources on|off, /feedback y|n, /exit",
)
```

Replace with:

```python
_print_header(console, vector_store_path)
_print_status(console, "Commands: /help, /sources on|off, /feedback y|n, /exit")
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_cli_visual.py -v
```

Expected: all tests pass including the 3 new header tests.

- [ ] **Step 6: Commit**

```bash
git add src/cli.py tests/test_cli_visual.py
git commit -m "feat: replace _print_header with branded bot banner"
```

---

## Task 4: Replace `_print_answer()` with inline `◈` format

**Files:**
- Modify: `src/cli.py`
- Modify: `tests/test_cli_visual.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_cli_visual.py`:

```python
def test_print_answer_uses_circle_mark(capsys):
    from src.cli import _print_answer
    _print_answer(None, "The answer is 42.")
    captured = capsys.readouterr()
    assert "◈" in captured.out


def test_print_answer_shows_bot_name(capsys):
    from src.cli import _print_answer
    _print_answer(None, "Some answer.")
    captured = capsys.readouterr()
    assert "Agentic Video RAG" in captured.out


def test_print_answer_shows_answer_text(capsys):
    from src.cli import _print_answer
    _print_answer(None, "Attention is all you need.")
    captured = capsys.readouterr()
    assert "Attention is all you need." in captured.out


def test_print_answer_no_panel_box(capsys):
    from src.cli import _print_answer
    _print_answer(None, "Test.")
    captured = capsys.readouterr()
    assert "╔" not in captured.out
    assert "╚" not in captured.out
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_cli_visual.py::test_print_answer_uses_circle_mark -v
```

Expected: `FAILED` — current `_print_answer` uses a Panel, not `◈`.

- [ ] **Step 3: Replace `_print_answer()` in `src/cli.py`**

Find and replace the entire `_print_answer` function:

```python
def _print_answer(console, answer: str):
    if console:
        console.print(f"\n[bold blue]◈ Agentic Video RAG[/bold blue]")
        for line in answer.splitlines():
            console.print(f"  {line}")
        console.print()
    else:
        print(f"\n◈ Agentic Video RAG")
        for line in answer.splitlines():
            print(f"  {line}")
        print()
```

- [ ] **Step 4: Remove unused `Panel` import**

At the top of `src/cli.py`, find:

```python
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
except ImportError:  # pragma: no cover - fallback for minimal installs
    Console = None
    Panel = None
    Table = None
```

Replace with:

```python
try:
    from rich.console import Console
    from rich.rule import Rule
    from rich.table import Table
except ImportError:  # pragma: no cover - fallback for minimal installs
    Console = None
    Rule = None
    Table = None
```

- [ ] **Step 5: Run all tests**

```bash
pytest tests/test_cli_visual.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/cli.py tests/test_cli_visual.py
git commit -m "feat: replace _print_answer Panel with inline ◈ response format"
```

---

## Task 5: Update `_print_status()` and the input prompt symbol

**Files:**
- Modify: `src/cli.py`
- Modify: `tests/test_cli_visual.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_cli_visual.py`:

```python
def test_print_status_has_dot_prefix(capsys):
    from src.cli import _print_status
    _print_status(None, "Loading store...")
    captured = capsys.readouterr()
    assert "·" in captured.out
    assert "Loading store..." in captured.out


def test_chat_prompt_uses_lightning_symbol():
    import inspect
    from src import cli
    source = inspect.getsource(cli.chat_command)
    assert "⚡" in source
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_cli_visual.py::test_print_status_has_dot_prefix tests/test_cli_visual.py::test_chat_prompt_uses_lightning_symbol -v
```

Expected: both `FAILED`.

- [ ] **Step 3: Replace `_print_status()` in `src/cli.py`**

Find and replace the entire `_print_status` function:

```python
def _print_status(console, message: str):
    if console:
        console.print(f"  [dim]· {message}[/dim]")
    else:
        print(f"  · {message}")
```

- [ ] **Step 4: Update the input prompt in `chat_command()`**

Find this line in `chat_command`:

```python
            query = input("\nYou> ").strip()
```

Replace with:

```python
            query = input("\n⚡ ").strip()
```

- [ ] **Step 5: Run all tests**

```bash
pytest tests/test_cli_visual.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/cli.py tests/test_cli_visual.py
git commit -m "feat: update status prefix to · and prompt to ⚡"
```

---

## Task 6: Fix Windows UTF-8 encoding in `main()`

**Files:**
- Modify: `src/cli.py`
- Modify: `tests/test_cli_visual.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_cli_visual.py`:

```python
def test_main_reconfigures_stdout_on_windows():
    import inspect
    from src import cli
    source = inspect.getsource(cli.main)
    assert "reconfigure" in source
    assert "utf-8" in source
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_cli_visual.py::test_main_reconfigures_stdout_on_windows -v
```

Expected: `FAILED`.

- [ ] **Step 3: Update `main()` in `src/cli.py`**

Find the current `main()` function:

```python
def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "command", None):
        args.command = "chat"
        args.func = chat_command
        if not hasattr(args, "show_sources"):
            args.show_sources = False

    args.func(args)
```

Replace with:

```python
def main(argv=None):
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "command", None):
        args.command = "chat"
        args.func = chat_command
        if not hasattr(args, "show_sources"):
            args.show_sources = False

    args.func(args)
```

- [ ] **Step 4: Run all tests**

```bash
pytest tests/test_cli_visual.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/cli.py tests/test_cli_visual.py
git commit -m "fix: reconfigure stdout/stderr to utf-8 on Windows for Unicode symbols"
```

---

## Task 7: Clean up temp file and final verification

**Files:**
- Delete: `_preview_bot.py`

- [ ] **Step 1: Delete the preview script**

```bash
git rm _preview_bot.py
```

- [ ] **Step 2: Run full test suite**

```bash
pytest tests/ -v
```

Expected: all tests pass, no errors.

- [ ] **Step 3: Smoke-test the CLI visually**

Run the chat command directly:

```bash
python -m src.cli chat --help
```

Expected: no crash, help text prints cleanly.

Then run the banner preview:

```bash
python -c "
import sys; sys.stdout.reconfigure(encoding='utf-8')
from src.cli import _print_header
_print_header(None, 'data/vector_store.pkl')
"
```

Expected: bot pixel art and branded header print in the terminal with colors.

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "chore: remove preview script, complete CLI visual overhaul"
```
