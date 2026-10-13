"""Exercise real CLI entry points with Windows legacy stream encodings."""

import json
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image


def test_json_cli_accepts_unicode_paths_under_legacy_encoding(tmp_path: Path) -> None:
    """JSON and file paths stay usable even when redirected stdout starts as cp1251."""
    image = tmp_path / "пожар-🛰.png"
    Image.new("RGB", (16, 16), (255, 140, 10)).save(image)
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "cp1251"
    environment["PYTHONUTF8"] = "0"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "satellite_analyzer",
            "analyze",
            str(image),
            "--output",
            str(tmp_path / "результаты-🛰"),
            "--json",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["image"] == image.name
    assert Path(report["artifacts"]["report"]).is_file()
    assert "Traceback" not in result.stderr


def test_human_cli_accepts_unicode_paths_under_legacy_encoding(tmp_path: Path) -> None:
    """Human summaries must finish after exporting under Unicode paths."""
    image = tmp_path / "visible.png"
    Image.new("RGB", (16, 16), (255, 140, 10)).save(image)
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "cp1251"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "satellite_analyzer",
            "analyze",
            str(image),
            "--output",
            str(tmp_path / "проверка-🛰"),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "Detected regions: 1" in result.stdout
    assert "проверка-🛰" in result.stdout
    assert "not a thermal satellite fire detection system" in " ".join(
        result.stdout.split()
    )
