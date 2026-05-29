import sys
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Pre-import app.bootstrap and app.service to ensure path resolutions and module attribute bindings are established for mock patches
import app.bootstrap
import app.service
from app.__main__ import async_main
from app.core.exceptions import ModelNotReadyError


def _find_printed_json(mock_print) -> dict:
    """Helper to search call history of mock_print for a valid JSON payload."""
    for call in mock_print.call_args_list:
        if not call[0]:
            continue
        first_arg = call[0][0]
        if not isinstance(first_arg, str):
            continue
        try:
            return json.loads(first_arg)
        except (json.JSONDecodeError, TypeError):
            continue
    raise AssertionError("No valid JSON payload printed to stdout in call history")


@pytest.mark.anyio
@patch("os.path.exists")
@patch("argparse.ArgumentParser.parse_args")
@patch("sys.exit")
@patch("builtins.print")
async def test_cli_missing_image(mock_print, mock_exit, mock_parse_args, mock_exists):
    """Test that CLI exits with 1 and prints an error JSON when the image doesn't exist."""
    args = MagicMock()
    args.imagePath = "non_existent_image.jpg"
    args.postId = "123"
    args.targetLang = "ENG"
    args.device = "cpu"
    args.verbose = False
    mock_parse_args.return_value = args

    mock_exists.return_value = False
    mock_exit.side_effect = SystemExit(1)

    with pytest.raises(SystemExit) as exc_info:
        await async_main()

    assert exc_info.value.code == 1
    mock_exit.assert_called_once_with(1)

    printed_json = _find_printed_json(mock_print)
    assert printed_json["postId"] == "123"
    assert printed_json["errorCode"] == "INVALID_INPUT"
    assert printed_json["status"] == "error"
    assert "Image file not found" in printed_json["error"]


@pytest.mark.anyio
@patch("os.path.exists")
@patch("argparse.ArgumentParser.parse_args")
@patch("sys.exit")
@patch("builtins.print")
@patch("app.service.TranslationPipelineCLI")
async def test_cli_success(mock_pipeline_class, mock_print, mock_exit, mock_parse_args, mock_exists):
    """Test that a successful pipeline execution prints the correct output JSON and exits with 0."""
    args = MagicMock()
    args.imagePath = "existing_image.png"
    args.postId = "456"
    args.targetLang = "ENG"
    args.device = "cpu"
    args.verbose = False
    mock_parse_args.return_value = args

    mock_exists.return_value = True
    mock_exit.side_effect = SystemExit(0)

    # Setup pipeline mock instance
    mock_pipeline = MagicMock()
    mock_pipeline.initialize = AsyncMock()
    mock_pipeline_success_response = {
        "postId": "456",
        "imagePath": "existing_image.png",
        "originalSize": {"width": 800, "height": 600},
        "translator": "deep-translator",
        "elapsedMs": 120,
        "timings": {},
        "overlays": [],
    }
    mock_pipeline.translate_image = AsyncMock(return_value=mock_pipeline_success_response)
    mock_pipeline_class.return_value = mock_pipeline

    with pytest.raises(SystemExit) as exc_info:
        await async_main()

    assert exc_info.value.code == 0
    mock_pipeline.initialize.assert_called_once()
    mock_pipeline.translate_image.assert_called_once_with(
        image_path="existing_image.png", post_id="456", target_lang="ENG"
    )
    mock_exit.assert_called_once_with(0)

    printed_json = _find_printed_json(mock_print)
    assert printed_json["postId"] == "456"
    assert printed_json["translator"] == "deep-translator"
    assert printed_json["imagePath"] == "existing_image.png"


@pytest.mark.anyio
@patch("os.path.exists")
@patch("argparse.ArgumentParser.parse_args")
@patch("sys.exit")
@patch("builtins.print")
@patch("app.service.TranslationPipelineCLI")
async def test_cli_sidecar_error(mock_pipeline_class, mock_print, mock_exit, mock_parse_args, mock_exists):
    """Test that raising a SidecarError prints a JSON error with its specific error code and exits with 1."""
    args = MagicMock()
    args.imagePath = "existing_image.png"
    args.postId = "789"
    args.targetLang = "ENG"
    args.device = "cpu"
    args.verbose = False
    mock_parse_args.return_value = args

    mock_exists.return_value = True
    mock_exit.side_effect = SystemExit(1)

    mock_pipeline = MagicMock()
    mock_pipeline.initialize = AsyncMock(side_effect=ModelNotReadyError("Model not loaded yet"))
    mock_pipeline_class.return_value = mock_pipeline

    with pytest.raises(SystemExit) as exc_info:
        await async_main()

    assert exc_info.value.code == 1
    mock_exit.assert_called_once_with(1)

    printed_json = _find_printed_json(mock_print)
    assert printed_json["postId"] == "789"
    assert printed_json["errorCode"] == "MODEL_NOT_READY"
    assert printed_json["status"] == "error"
    assert "Model not loaded yet" in printed_json["error"]


@pytest.mark.anyio
@patch("os.path.exists")
@patch("argparse.ArgumentParser.parse_args")
@patch("sys.exit")
@patch("builtins.print")
@patch("app.service.TranslationPipelineCLI")
async def test_cli_unexpected_exception(mock_pipeline_class, mock_print, mock_exit, mock_parse_args, mock_exists):
    """Test that raising an unexpected native Exception prints an INTERNAL_ERROR JSON and exits with 1."""
    args = MagicMock()
    args.imagePath = "existing_image.png"
    args.postId = "999"
    args.targetLang = "ENG"
    args.device = "cpu"
    args.verbose = False
    mock_parse_args.return_value = args

    mock_exists.return_value = True
    mock_exit.side_effect = SystemExit(1)

    mock_pipeline = MagicMock()
    mock_pipeline.initialize = AsyncMock(side_effect=RuntimeError("Some hardware failure!"))
    mock_pipeline_class.return_value = mock_pipeline

    with pytest.raises(SystemExit) as exc_info:
        await async_main()

    assert exc_info.value.code == 1
    mock_exit.assert_called_once_with(1)

    printed_json = _find_printed_json(mock_print)
    assert printed_json["postId"] == "999"
    assert printed_json["errorCode"] == "INTERNAL_ERROR"
    assert printed_json["status"] == "error"
    assert "Some hardware failure!" in printed_json["error"]
