# Deployment Guide

## Streamlit Cloud

1. **Fork / push** the repository to your GitHub account.
2. Visit [share.streamlit.io](https://share.streamlit.io) and select **New app**.
3. Choose your repo, branch, and set the main file to `scripts/lcf_receipt_entry_streamlit.py`.
4. Add the following files to the repo root (already present in this project):
   - `requirements.txt`
   - `pyproject.toml`
   - `packages.txt` (installs system dependencies like `poppler-utils`, `libGL`, etc.)
5. Set environment secrets (Settings → Secrets):
   ```toml
   OPENAI_API_KEY="..."
   GEMINI_API_KEY="..."
   GOOGLE_CREDENTIALS_JSON="..."  # optional; use file uploader when possible
   ```
   > The hosted instance still supports uploading credentials via the sidebar; secrets only need to be set if you prefer environment variables.
6. Deploy. Streamlit Cloud caches wheels, so the second deploy is significantly faster.

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
cd receipt_scanner
python -m venv scanner-venv
scanner-venv\Scripts\activate  # or source scanner-venv/bin/activate
pip install -r requirements.txt
streamlit run scripts/lcf_receipt_entry_streamlit.py
```

### Environment Variables

If you cannot use the in-app settings, set paths/keys via `.env` or shell variables:

```
OPENAI_API_KEY=...
GEMINI_API_KEY=...
GOOGLE_APPLICATION_CREDENTIALS=/abs/path/to/creds.json
```

### Headless Tips

- Disable `st.file_uploader` and feed receipts programmatically via CLI if needed.
- Use `config.txt` under `scannerai/_config/` to mirror the settings JSON.
- For GPU instances, ensure the appropriate CUDA libraries are installed (OpenCV works fine on CPU for this workflow).

## Updating the Hosted App

1. Commit changes to `main`.
2. Push to GitHub.
3. Streamlit Cloud auto-pulls from `main`. Use the **Rerun** button in the dashboard if the deploy does not pick up immediately.

## Monitoring & Logs

- Streamlit Cloud surfaces logs in the browser (upper-right → **Manage app** → **Logs**).
- For self-hosted deployments, use `streamlit run ... --server.port=... --server.headless=true` and capture stdout/stderr with your preferred process manager.

