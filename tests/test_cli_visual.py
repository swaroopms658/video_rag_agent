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
