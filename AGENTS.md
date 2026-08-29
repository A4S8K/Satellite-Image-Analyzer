# Development Guide

- Target Python 3.12 or newer and keep public APIs fully typed.
- Keep Week 1 focused on image I/O, metadata, preprocessing, and the CLI.
- Run `pytest`, `ruff check .`, `black --check .`, and `mypy src` before handoff.
- Use synthetic images in tests; do not add large imagery or external data dependencies.
