# Deployment Guide

This guide covers **hosted deployments** (Streamlit Cloud) and **self-hosted** Streamlit servers. If you only need the packaged Windows desktop app, download it from the [local pre-release](https://github.com/rijff24/receiptAI/releases/tag/Alpha0.1.0) or build it via [`WINDOWS_INSTALL.md`](WINDOWS_INSTALL.md).

## Streamlit Cloud

1. **Fork / push** the repository to your GitHub account.
2. Visit [share.streamlit.io](https://share.streamlit.io) and select **New app**.
3. Choose your repo, branch, and set the main file to `scripts/lcf_receipt_entry_streamlit.py`.
4. Use `main` for production deployments and `dev` when smoke-testing upcoming changes.
5. Add the following files to the repo root (already present in this project):
   - `requirements.txt`
   - `pyproject.toml`
   - `packages.txt` (installs system dependencies like `poppler-utils`, `libGL`, etc.)
6. Decide how credentials should be supplied:
   - Recommended hosted mode: users enter their own API keys in **Application Settings**, then export/import `scannerai_settings.json` for reuse.
   - Operator-managed fallback: set OS environment variables for centrally managed deployments.
   ```toml
   OPENAI_API_KEY="..."
   GEMINI_API_KEY="..."
   GOOGLE_CREDENTIALS_PATH="/mounted/path/to/google-credentials.json"
   ```
   > The current code reads environment variables through `os.environ`. If you use Streamlit Cloud secrets, make sure they are exposed as environment variables or enter keys through the UI.
7. Deploy. Streamlit Cloud caches wheels, so the second deploy is significantly faster.

### Common Issues

| Symptom | Fix |
| ------- | --- |
| `ModuleNotFoundError: scannerai` | Ensure `pyproject.toml` has `setuptools.packages.find` pointing to `src`, and `requirements.txt` installs the package via `-e .`. |
| `ImportError: libGL.so.1` | Confirm `packages.txt` includes `poppler-utils` and `libgl1`. Already handled in this repo. |
| `ValueError: project.authors[0].email` | Remove invalid email fields from `pyproject.toml`. Already resolved here. |
| `tkinter` errors | The UI no longer depends on Tk; make sure you run the latest code. |

## Self-Hosted (Streamlit)

```bash
git clone https://github.com/rijff24/receiptAI.git
cd receiptAI
python -m venv scanner-venv
scanner-venv\Scripts\activate  # or source scanner-venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
streamlit run scripts/lcf_receipt_entry_streamlit.py
```

Set `SCANNERAI_HOSTED_MODE=0` when you want persistent local settings on the server. Leave it as `1` for stateless hosted behavior.

### Environment Variables

If you cannot use the in-app settings, set paths/keys via `.env` or shell variables:

```
SCANNERAI_HOSTED_MODE=0
OPENAI_API_KEY=...
GEMINI_API_KEY=...
OPENAI_API_KEY_PATH=/abs/path/to/openai.key
GEMINI_API_KEY_PATH=/abs/path/to/gemini.key
GOOGLE_CREDENTIALS_PATH=/abs/path/to/google-credentials.json
```

### Headless Tips

- For non-interactive automation, reuse the OCR processor classes directly instead of the Streamlit UI.
- Use `config.txt` under `src/scannerai/_config/` to mirror the settings JSON.
- For GPU instances, ensure the appropriate CUDA libraries are installed (OpenCV works fine on CPU for this workflow).

## Updating the Hosted App

1. Commit changes to `main`.
2. Push to GitHub.
3. Streamlit Cloud auto-pulls from `main`. Use the **Rerun** button in the dashboard if the deploy does not pick up immediately.

## Monitoring & Logs

- Streamlit Cloud surfaces logs in the browser (upper-right → **Manage app** → **Logs**).
- For self-hosted deployments, use `streamlit run ... --server.port=... --server.headless=true` and capture stdout/stderr with your preferred process manager.

