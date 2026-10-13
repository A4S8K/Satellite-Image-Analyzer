"""Strict JSON reporting; image arrays are deliberately not serialized."""

import json
from pathlib import Path
from typing import Any

from satellite_analyzer.models import AnalysisResult


def report_payload(
    result: AnalysisResult, artifacts: dict[str, str] | None = None
) -> dict[str, Any]:
    """Create one consistent payload for the saved report and JSON stdout."""
    payload = result.to_dict()
    if artifacts is not None:
        payload["artifacts"] = artifacts
    return payload


def write_json_report(
    result: AnalysisResult, path: Path, artifacts: dict[str, str] | None = None
) -> None:
    """Write a new UTF-8 report, rejecting nonfinite values and existing files."""
    serialized = json.dumps(
        report_payload(result, artifacts), indent=2, ensure_ascii=False, allow_nan=False
    )
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized + "\n")
