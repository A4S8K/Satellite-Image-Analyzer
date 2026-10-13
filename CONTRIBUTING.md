# Contributing

Use Python 3.12 or newer and read [docs/architecture.md](docs/architecture.md).
Week 2 covers visible RGB fire-like detection; Week 3 adds calibrated Red/NIR
vegetation analysis and georeferenced NDVI. Risk-engine work remains planned.

## Setup and checks

From a repository checkout in PowerShell:

```powershell
$env:PYTHONUTF8 = '1'
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m black --check .
.\.venv\Scripts\python.exe -m mypy src
```

Activation is optional. UTF-8 mode supports repository paths containing non-ASCII
characters when packaging with Windows Python 3.12. To format files, run
`python -m black .` with the environment's Python and inspect the diff.

## Implementation conventions

- Keep public APIs typed, functions small, and non-obvious decisions documented.
- Use `pathlib` for paths and logging for internal diagnostics.
- Put adjustable thresholds in validated configuration, not duplicated constants.
- Preserve source pixels and write artifacts in a new run directory.
- Keep detector results compatible with the reporting/visualization contract.
- Distinguish heuristic confidence from calibrated probabilities and accuracy.
- Do not claim RGB thresholds measure thermal hotspots or true NDVI.
- Add optional model dependencies only alongside implementations that need them.

## Tests and imagery

Use small deterministic synthetic images and temporary directories. Verify
distractor colors, filtering, empty detections, coordinate scaling, invalid inputs,
and CLI/export failures. Do not download tests' imagery or commit large rasters.
Synthetic fixtures show software behavior, not scientific validation.

For any licensed sample, record source, terms, sensor, bands, and preparation in
`data/README.md`. Evaluate accuracy against independent labeled satellite imagery.

## Pull requests

Explain the user-visible behavior, algorithm/data assumptions, and checks run.
Include a screenshot for visualization changes and note report/schema changes.
Keep unrelated edits out and remove merge conflict markers from committed files.
