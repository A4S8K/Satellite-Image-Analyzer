"""Tests for the public command-line interface."""

from pathlib import Path

from typer.testing import CliRunner

from satellite_analyzer.cli import app

runner = CliRunner()


def test_help() -> None:
    """The CLI exposes discoverable help."""

    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Inspect and preprocess satellite image files" in result.stdout


def test_version() -> None:
    """The version command succeeds."""

    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert "Satellite Image Analyzer 0.1.0" in result.stdout


def test_info(png_path: Path) -> None:
    """The info command renders loaded image metadata."""

    result = runner.invoke(app, ["info", str(png_path)])

    assert result.exit_code == 0
    assert "sample.png" in result.stdout
    assert "Width" in result.stdout


def test_preprocess(png_path: Path, tmp_path: Path) -> None:
    """The preprocess command writes an output image."""

    output_path = tmp_path / "processed.png"
    result = runner.invoke(
        app,
        [
            "preprocess",
            str(png_path),
            "--width",
            "12",
            "--height",
            "8",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert output_path.is_file()
    assert "Saved processed image" in result.stdout
