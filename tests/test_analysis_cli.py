"""End-to-end fire analysis CLI tests using deterministic local images."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from satellite_analyzer.cli import app

runner = CliRunner()


@pytest.fixture()
def fire_image(tmp_path: Path) -> Path:
    """Make two separated warm regions on a green, non-fire background."""
    rgb = np.empty((64, 80, 3), dtype=np.uint8)
    rgb[:] = (35, 130, 40)
    rgb[10:22, 12:28] = (255, 140, 20)
    rgb[36:44, 52:62] = (255, 235, 20)
    path = tmp_path / "visible fire image.png"
    Image.fromarray(rgb).save(path)
    return path


def _json_run(image: Path, output: Path, *options: str) -> dict[str, Any]:
    """Require successful, strictly JSON stdout before inspecting report fields."""
    result = runner.invoke(
        app, ["analyze", str(image), "--output", str(output), "--json", *options]
    )
    assert result.exit_code == 0, result.output
    payload: dict[str, Any] = json.loads(result.stdout)
    return payload


def test_analysis_json_and_all_artifacts(fire_image: Path, tmp_path: Path) -> None:
    """Export masks and figures that agree with the report without changing source."""
    before = fire_image.read_bytes()
    output = tmp_path / "analysis output"
    report = _json_run(fire_image, output, "--verbose", "--save-overlay")

    assert report["analysis_scope"] == "fire_only_week_2"
    assert report["resolution"] == {"width": 80, "height": 64}
    assert report["analysis_resolution"] == report["resolution"]
    assert report["fire"]["pixel_count"] == 272
    assert report["fire"]["percentage"] == pytest.approx(272 * 100 / 5120)
    assert report["fire"]["region_count"] == 2
    assert 0 < report["fire"]["confidence"] <= 1
    assert report["unknown"]["pixel_count"] == 5120 - 272
    assert report["unknown"]["percentage"] + report["fire"]["percentage"] == 100
    assert report["invalid_pixel_count"] == 0
    assert report["processing_time_seconds"] >= 0
    assert not {"vegetation", "water", "urban"}.intersection(report)

    artifact_paths = {
        Path(value).name: Path(value) for value in report["artifacts"].values()
    }
    assert set(artifact_paths) == {
        "analyzed_image.png",
        "fire_mask.png",
        "fire_overlay.png",
        "analysis_summary.png",
        "analysis_report.json",
    }
    runs = list(output.iterdir())
    assert len(runs) == 1 and runs[0].is_dir()
    assert all(
        path.is_absolute() and path.parent == runs[0]
        for path in artifact_paths.values()
    )
    saved_report = json.loads(
        artifact_paths["analysis_report.json"].read_text(encoding="utf-8")
    )
    assert saved_report == report
    for name in ("analyzed_image.png", "fire_overlay.png"):
        with Image.open(artifact_paths[name]) as image:
            image.load()
            assert image.size == (80, 64)
            assert image.mode == "RGB"
    with Image.open(artifact_paths["analysis_summary.png"]) as image:
        image.load()
        assert image.width > 80 and image.height > 64
    with Image.open(artifact_paths["fire_mask.png"]) as image:
        mask = np.asarray(image)
        assert image.size == (80, 64)
        assert image.mode == "L"
    assert set(np.unique(mask)) == {0, 255}
    assert np.count_nonzero(mask) == report["fire"]["pixel_count"]
    assert all(
        region["pixel_count"] > 0
        and np.count_nonzero(
            mask[
                region["y"] : region["y"] + region["height"],
                region["x"] : region["x"] + region["width"],
            ]
        )
        == region["pixel_count"]
        for region in report["fire"]["regions"]
    )
    assert fire_image.read_bytes() == before


def test_analysis_keeps_previous_run(fire_image: Path, tmp_path: Path) -> None:
    """A repeat analysis gets a new run directory and preserves the first report."""
    output = tmp_path / "results"
    first = _json_run(fire_image, output)
    first_report = Path(first["artifacts"]["report"])
    before = first_report.read_bytes()

    second = _json_run(fire_image, output)

    assert first_report != Path(second["artifacts"]["report"])
    assert len(list(output.iterdir())) == 2
    assert first_report.read_bytes() == before


def test_analysis_no_fire(tmp_path: Path) -> None:
    """A green image yields an empty valid mask, no regions, and zero confidence."""
    image = tmp_path / "no_fire.png"
    Image.new("RGB", (48, 32), (20, 120, 35)).save(image)

    report = _json_run(image, tmp_path / "results")

    assert report["fire"] == {
        "pixel_count": 0,
        "percentage": 0.0,
        "confidence": 0.0,
        "region_count": 0,
        "regions": [],
    }
    assert report["unknown"] == {"pixel_count": 48 * 32, "percentage": 100.0}
    with Image.open(report["artifacts"]["fire_mask"]) as mask:
        assert not np.asarray(mask).any()


def test_analysis_invalid_config(fire_image: Path, tmp_path: Path) -> None:
    """Invalid thresholds fail on stderr before artifacts are created."""
    config = tmp_path / "invalid.toml"
    config.write_text("[fire]\nmin_region_pixels = 0\n", encoding="utf-8")
    output = tmp_path / "results"

    result = runner.invoke(
        app,
        [
            "analyze",
            str(fire_image),
            "--config",
            str(config),
            "--output",
            str(output),
            "--json",
        ],
    )

    assert result.exit_code != 0
    assert result.stdout == ""
    assert "Error:" in result.stderr
    assert "min_region_pixels" in result.stderr
    assert "Traceback" not in result.stderr
    assert not output.exists()


def test_analysis_custom_config(fire_image: Path, tmp_path: Path) -> None:
    """The CLI actually applies configured component filtering to detector output."""
    config = tmp_path / "custom.toml"
    config.write_text("[fire]\nmin_region_pixels = 150\n", encoding="utf-8")

    report = _json_run(fire_image, tmp_path / "results", "--config", str(config))

    assert report["fire"]["pixel_count"] == 192
    assert report["fire"]["region_count"] == 1
    assert report["parameters"]["fire"]["min_region_pixels"] == 150


def test_analysis_missing_input(tmp_path: Path) -> None:
    """A missing image produces a clear error and does not create an output run."""
    output = tmp_path / "results"

    result = runner.invoke(
        app,
        ["analyze", str(tmp_path / "missing.png"), "--output", str(output), "--json"],
    )

    assert result.exit_code != 0
    assert result.stdout == ""
    assert "Error:" in result.stderr
    assert "Traceback" not in result.stderr
    assert not output.exists()


def test_analysis_output_root_is_file(fire_image: Path, tmp_path: Path) -> None:
    """Reject an output file without overwriting its contents or hiding the error."""
    output = tmp_path / "occupied.txt"
    output.write_text("preserve existing user content", encoding="utf-8")
    before = output.read_bytes()

    result = runner.invoke(
        app, ["analyze", str(fire_image), "--output", str(output), "--json"]
    )

    assert result.exit_code != 0
    assert result.stdout == ""
    assert "Error:" in result.stderr
    assert "Traceback" not in result.stderr
    assert output.read_bytes() == before
