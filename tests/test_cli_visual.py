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
