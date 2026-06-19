# Troubleshooting Guide

## Installation

| Issue | Fix |
| ----- | --- |
| `pip` fails on Windows because of `cryptography` | Ensure you are on Python 3.11+ and upgrade `pip`, `setuptools`, `wheel`. |
| `ModuleNotFoundError: scannerai` | Run `pip install -e .` or install via `requirements.txt` (which already includes `-e .`). |
| `tkinter` import errors | The UI no longer depends on Tk—update to the latest `main`. |

## Local Desktop (ScannerAI.exe)

- **Windows SmartScreen warning** – The alpha build is unsigned. Click **More info → Run anyway** if you trust the binary from the [official release](https://github.com/rijff24/receiptAI/releases/tag/Alpha0.1.0).
- **Long first launch** – PyInstaller extracts dependencies on first run; subsequent launches are faster. Keep the console window open until the browser loads `http://localhost:8501`.
- **Settings not persisting** – The Windows launcher sets `SCANNERAI_HOSTED_MODE=0`, so settings live under `%APPDATA%\ScannerAI`. If you run `streamlit run scripts/lcf_receipt_entry_streamlit.py` from source, set `SCANNERAI_HOSTED_MODE=0` yourself; otherwise the `main` branch defaults to hosted/session-only mode.

## Settings & API Keys

- **Settings form missing** – Install `cryptography` + `keyring` (`pip install -r requirements.txt`).
- **Keys not saving** – Check that `user_settings.json` exists and is writable (see `SETTINGS.md` for locations).
- **Encryption key missing** – Delete `.scannerai.key` and re-enter settings; it will regenerate.

## OCR Providers

- **Gemini** – Requires Google service-account JSON and Gemini API key. Set `GOOGLE_CREDENTIALS_PATH`, enter the path in **Application Settings**, or upload the JSON via the local-mode UI.
- **GPT-4 Vision** – Needs OpenAI API key with vision access.
- **Tesseract + GPT-3.5** – Install Tesseract locally and set `tesseract_cmd_path`.

## Streamlit UI

- **Buttons disabled after processing** – Hit the **Home** button to clear state, or upload new files. If using a hosted instance, refresh the page to reset `session_state`.
- **Images not rendering** – Confirm receipts are RGB. PDFs are converted via `pdf2image`; ensure `poppler-utils` is installed (see `packages.txt`).
- **Browse button missing** – Ensure you are not running in a very small viewport; Streamlit collapses widgets on mobile.

## Export

- **CSV empty** – Make sure receipts were processed (progress bar should show Completed n/n). Items export only when item capture is enabled.
- **JSON has no `items` key** – Items are only included when the setting **Enable item capture and editing** is on.

## Deployment

See `DEPLOYMENT.md` for Streamlit Cloud-specific tips. Common problems include missing system packages, forgetting `-e .` installs, or invalid `pyproject.toml` metadata.

## Need Help?

- Open a GitHub Issue with logs, screenshots, and reproduction steps.
- Security reports should go to `rijff@swan-computing.com`.

