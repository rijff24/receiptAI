"""Launcher script for running ScannerAI in local (desktop) mode."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from streamlit.web import cli as stcli


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    app_script = repo_root / "scripts" / "lcf_receipt_entry_streamlit.py"

    if not app_script.exists():
        print(f"Unable to locate Streamlit app at: {app_script}", file=sys.stderr)
        return 1

    # Ensure local mode and disable telemetry prompts for bundled builds.
    os.environ.setdefault("SCANNERAI_HOSTED_MODE", "0")
    os.environ.setdefault("SCANNERAI_LOCAL_LAUNCHER", "1")
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "0")

    # Delegate to Streamlit's CLI entrypoint.
    sys.argv = ["streamlit", "run", str(app_script)]
    stcli.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

