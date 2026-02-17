# ScannerAI Windows Packaging Guide

This guide describes how to build a Windows-friendly executable for ScannerAI. The executable simply launches the existing Streamlit UI in your default browser, but ships with an embedded Python environment so end users do not need to install Python manually.

> **For end users:** The latest packaged build lives on the GitHub release page: [ScannerAI Local Desktop - Alpha Pre-Release (v0.1.0-local-alpha)](https://github.com/rijff24/receiptAI/releases/tag/Alpha0.1.0). Download `ScannerAI.exe` from there unless you specifically need to rebuild the launcher.

## For Developers

This document targets contributors working on the **local desktop** experience (branches `local-dev` and `local-main`). Follow it when you need to regenerate `ScannerAI.exe`, test PyInstaller changes, or publish a new pre-release.

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

1. From the repo root, ensure dependencies are installed and the launcher runs:

   ```powershell
   .\scanner-venv\Scripts\Activate.ps1
   python launch_scannerai.py --help   # should start Streamlit locally
   ```

2. **Close any running ScannerAI.exe or Streamlit** before building. If the EXE is in use, PyInstaller will fail with `PermissionError: Access is denied` when writing `dist\ScannerAI.exe`.

3. Run PyInstaller using the **spec file** (recommended; it includes all required hidden imports and data):

   ```powershell
   pyinstaller --noconfirm --clean ScannerAI.spec
   ```

   The spec file `ScannerAI.spec`:
   - Uses `collect_all` for `streamlit`, `cv2`, and `scannerai`.
   - Declares **explicit hidden imports** for scannerai dependencies that PyInstaller does not always trace (see below).
   - Produces a single-file EXE in `dist/ScannerAI.exe`.

4. **Smoke-test the EXE** after every build. Run `dist\ScannerAI.exe` and confirm the app loads in the browser at http://localhost:8501 with no `ModuleNotFoundError`. If you see a missing module, add it to the hidden imports in `ScannerAI.spec` and rebuild.

5. The first run may show a Windows SmartScreen prompt; code-signing is recommended for wider distribution (future enhancement). Tag alpha builds (e.g., `v0.1.0-local-alpha`) as **pre-releases** and upload the EXE to GitHub.

6. (Optional) Wrap the `dist` output in an installer (e.g., WiX, Inno Setup) if you want a guided installation experience.

### Hidden imports (ScannerAI.spec)

PyInstaller does not always trace imports that are used only inside the `scannerai` package or loaded lazily. If any of these are missing from the spec, the EXE will raise `ModuleNotFoundError` at runtime. The spec file must list them explicitly in the `hiddenimports` section:

| Module | Used by | Purpose |
|--------|---------|---------|
| `dotenv` | `scannerai._config.config` | python-dotenv for config file loading |
| `pdf2image` | `scannerai.utils.scanner_utils` | PDF to image conversion |
| `tiktoken` | `scannerai.utils.scanner_utils` | Token counting |
| `openai` | `scannerai.ocr.lcf_receipt_process_gpt4vision`, `lcf_receipt_process_openai` | OpenAI API client |
| `pytesseract` | `scannerai.ocr.lcf_receipt_process_openai` | Tesseract OCR wrapper |
| `google.generativeai` | `scannerai.ocr.lcf_receipt_process_gemini` | Gemini API client |

**When adding new scannerai code** that `import` or `from` a third-party package, add that package’s **import name** (e.g. `openai`, not `openai-api`) to the `hiddenimports` list in `ScannerAI.spec`, then rebuild and run the EXE to verify.

## Runtime behavior

- The launcher sets `SCANNERAI_HOSTED_MODE=0` so all settings live under the user's profile (see `SETTINGS.md`).
- When the EXE starts, it spawns Streamlit inside the bundled environment and opens the default browser at `http://localhost:8501`.
- Use the new **Cancel Processing** and **Exit Application** controls in the sidebar to stop batch jobs or end the session cleanly.

## Updating the build

1. Make changes on `local-dev`.
2. If you added or changed imports in `scannerai` or the Streamlit script, update `ScannerAI.spec` hidden imports if needed (see table above).
3. Close any running ScannerAI.exe, then run `pyinstaller --noconfirm --clean ScannerAI.spec`.
4. Run `dist\ScannerAI.exe` and confirm the app loads without `ModuleNotFoundError`.
5. Merge `local-dev` → `local-main` when ready; tag a release for distribution.

For hosted updates, continue using `cloud-dev` → `main`.

