"""Portable UTF-8 output for Windows console and redirected CLI streams."""

import sys


def configure_cli_streams() -> None:
    """Emit UTF-8 even when Windows gives redirected streams an ANSI code page.

    Paths can contain Cyrillic or emoji. Reconfigure real text streams at the
    CLI entry point, leaving StringIO/test/application-owned streams untouched.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="backslashreplace")
