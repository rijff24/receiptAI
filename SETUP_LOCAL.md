# Local setup (this machine)

## Done for you

- **Repository**: Code from `https://github.com/rijff24/receiptAI.git` is in this folder. This checkout is currently on `main`; `dev` is also available and tracks `origin/dev`.
- **Virtual environment**: `scanner-venv` uses **Python 3.11** (required: scipy/scikit-learn have no Windows wheels for Python 3.14; 3.11 avoids Fortran build errors).
- **Dependencies**: Installed in `scanner-venv`; `requirements.txt` is the Streamlit runtime entrypoint and `pyproject.toml` contains the package metadata.

## One-time: install dependencies (if starting from scratch)

Use **Python 3.11** (not 3.14) so scipy/scikit-learn install from wheels. With the py launcher:

```powershell
cd c:\dev\receiptAI_v1
py -3.11 -m venv scanner-venv
.\scanner-venv\Scripts\Activate.ps1
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

If `pip install -r requirements.txt` is slow or fails, the resolver may backtrack; using Python 3.11 avoids the scipy Fortran build error.

If you do development (linting, pre-commit):

```powershell
pip install -e ".[dev]"
pre-commit install
```

## Run the app

```powershell
cd c:\dev\receiptAI_v1
.\scanner-venv\Scripts\Activate.ps1
$env:SCANNERAI_HOSTED_MODE="0"
streamlit run scripts/lcf_receipt_entry_streamlit.py
```

Then open the URL shown (usually http://localhost:8501). Configure OCR and API keys in **Application Settings** in the sidebar.

## Git

- **On dev**: `git pull` then `git push origin dev`
- **On main**: `git pull` then `git push origin main`
- **Merge dev → main** (when dev is ready): `git checkout main`, `git merge dev`, then `git push origin main`
