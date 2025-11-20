# ScannerAI Windows Packaging Guide

This guide describes how to build a Windows-friendly executable for ScannerAI. The executable simply launches the existing Streamlit UI in your default browser, but ships with an embedded Python environment so end users do not need to install Python manually.

## Branch workflow

- `main`: hosted (cloud) production.
- `cloud-dev`: upcoming hosted features before they land on `main`.
- `local-dev`: active work on the Windows build/installer.
- `local-main`: stable branch for published Windows releases.

The launcher script (`launch_scannerai.py`) and packaging configuration live on the local branches.

## Prerequisites

- Windows 10/11 64-bit.
- Python 3.11.x installed (match the version used in `scanner-venv`).
- All project dependencies installed: `pip install -r requirements.txt`.
- PyInstaller: `pip install pyinstaller`.

> Optional: run inside `scanner-venv` (`.\scanner-venv\Scripts\activate`) so the build uses the same environment as local development.

## Build steps

1. From the repo root (`ScannerAI/receipt_scanner`), ensure dependencies are installed and the launcher is available:

   ```powershell
   python launch_scannerai.py --help  # should start Streamlit locally
   ```

2. Run PyInstaller (one-file build with collected package assets):

   ```powershell
   pyinstaller ^
     --noconfirm ^
     --clean ^
     --name ScannerAI ^
     --collect-all scannerai ^
     --add-data "scripts/lcf_receipt_entry_streamlit.py;scripts" ^
     launch_scannerai.py
   ```

   Key flags:
   - `--collect-all scannerai` ensures the package data (OCR models, settings schema) ships with the EXE.
   - `--add-data` bundles the Streamlit entry script relative to the executable.

3. The build artifacts land in `dist/ScannerAI/`. `ScannerAI.exe` can be double-clicked or distributed as-is. The first run may show a Windows SmartScreen prompt; code-signing is recommended for wider distribution (future enhancement).

4. (Optional) Wrap the `dist/ScannerAI` folder in an installer (e.g., WiX, Inno Setup) if you want a guided installation experience.

## Runtime behavior

- The launcher sets `SCANNERAI_HOSTED_MODE=0` so all settings live under the user's profile (see `SETTINGS.md`).
- When the EXE starts, it spawns Streamlit inside the bundled environment and opens the default browser at `http://localhost:8501`.
- Use the new **Cancel Processing** and **Exit Application** controls in the sidebar to stop batch jobs or end the session cleanly.

## Updating the build

1. Make changes on `local-dev`.
2. Run the PyInstaller command above; smoke-test the EXE.
3. Merge `local-dev` → `local-main` when ready; tag a release for distribution.

For hosted updates, continue using `cloud-dev` → `main`.

