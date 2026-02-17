# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_all
from PyInstaller.utils.hooks import copy_metadata

datas = [('scripts/lcf_receipt_entry_streamlit.py', 'scripts')]
binaries = []
hiddenimports = []

# Collect all Streamlit files including static assets
tmp_ret = collect_all('streamlit')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# Collect OpenCV (cv2) - required for image processing
tmp_ret = collect_all('cv2')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# Collect metadata for other packages
datas += copy_metadata('altair')
datas += copy_metadata('watchdog')

# Collect scannerai package
tmp_ret = collect_all('scannerai')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# Explicit hidden imports for scannerai dependencies (not always traced by PyInstaller).
# If the EXE raises ModuleNotFoundError at runtime, add the missing module here and rebuild.
# See WINDOWS_INSTALL.md "Hidden imports" section for the full list and when to add new ones.
hiddenimports += [
    'dotenv',           # scannerai._config.config (python-dotenv)
    'pdf2image',        # scannerai.utils.scanner_utils
    'tiktoken',         # scannerai.utils.scanner_utils
    'openai',           # scannerai.ocr (GPT-4 Vision, OpenAI OCR)
    'pytesseract',      # scannerai.ocr.lcf_receipt_process_openai
    'google.generativeai',  # scannerai.ocr.lcf_receipt_process_gemini
]

a = Analysis(
    ['launch_scannerai.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Development and testing tools (safe to exclude)
        'pytest', 'pytest_cov', 'pytest_randomly', 'hypothesis',
        'pylint', 'astroid', 'isort', 'ruff', 'pre_commit',
        
        # Jupyter/IPython (not needed for runtime)
        'IPython', 'ipykernel', 'jupyter', 'notebook', 'jupyter_client',
        'jupyter_core', 'jupyterlab', 'ipython',
        
        # Tkinter (already confirmed not needed)
        'tkinter', '_tkinter',
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='ScannerAI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    exclude_binaries=False,  # Single-file executable
    icon=r'ReceiptAI.ico',  # Custom icon for the executable (relative to spec file location)
)
# COLLECT section removed - not needed for single-file executable
