# ScannerAI Architecture Guide

## High-Level Flow

1. **Upload** – Users upload images/PDFs through the Streamlit sidebar (`scripts/lcf_receipt_entry_streamlit.py`).
2. **Processing Queue** – Files are queued and processed sequentially by a background worker, updating Streamlit `session_state`.
3. **OCR Layer** (`src/scannerai/ocr/`) – Receipts are run through the selected OCR provider (Gemini Vision, GPT-4 Vision via GPT-4o mini, or Tesseract + GPT-3.5).
4. **Classification Layer** (`src/scannerai/classifiers/`) – Optional COICOP classifier enriches the results.
5. **Settings & Secrets** (`src/scannerai/settings/`) – UI changes persist locally with encrypted keys in local mode, or remain session-only with encrypted export/import in hosted mode.
6. **UI + Editing** – Users review results, adjust metadata/items/VAT/notes, retry or skip failures, delete receipts, and export a ZIP with JSON/CSV plus renamed uploads.

```
Uploads -> Processing Queue -> OCR Processor -> Classification -> Session State -> UI -> Export
```

## Module Overview

| Path | Responsibility |
| ---- | -------------- |
| `scripts/lcf_receipt_entry_streamlit.py` | Streamlit UI, uploads, session state, export |
| `src/scannerai/ocr/` | OCR engines (`lcf_receipt_process_gemini.py`, `lcf_receipt_process_gpt4vision.py`, `lcf_receipt_process_openai.py`) |
| `src/scannerai/classifiers/` | COICOP classifier (model loading, inference helpers) |
| `src/scannerai/settings/` | `SettingsManager` (encryption, keyring, default paths, hosted export/import) |
| `src/scannerai/_config/` | Legacy config reader for headless usage |
| `src/scannerai/utils/` | Helpers (PDF merging, token counts, etc.) |

## Settings & Secrets Flow

1. User edits values in **Application Settings**.
2. `SettingsManager` validates input. In local mode it saves under `user_settings.json`; in hosted mode it keeps data in memory until the user downloads an encrypted export.
3. API keys are encrypted using `cryptography.Fernet`. Encryption key is stored in the OS keyring when possible.
4. OCR processors read settings each time a file is processed, so changing keys or toggles takes effect immediately.

## Session State Model

`st.session_state` keys used in the UI:

- `results`: list of processed receipts (`{"image": ..., "receipt_data": ...}`)
- `current_index`: index of the receipt being reviewed
- `receipt_status`: status tracker for each uploaded file
- `failed_receipts`: files that errored and can be retried or skipped
- `processing_queue`: pending files
- `processing_active`: boolean gate for background work
- `processing_payloads`: uploaded file bytes retained for retry/error handling
- `process_counts`: completed/total counters for progress display
- `file_uploader_key`: rotates to clear uploads when hitting Home
- `autosave_path`: JSON autosave target, defaulting to `receipt_autosave.json`

## Extending OCR Providers

1. Create `src/scannerai/ocr/lcf_receipt_process_newprovider.py` implementing `process_receipt`.
2. Register the provider in `scripts/lcf_receipt_entry_streamlit.py` within the OCR initialization block.
3. Add settings form inputs if the provider needs API keys or custom config.
4. Update docs: README (Features + Usage) and ARCHITECTURE.

## Extending Classification

1. Add your model + artifacts under `scannerai/classifiers/`.
2. Provide a loader/inference helper similar to `lcf_classify.py`.
3. Wire the classifier into the Streamlit workflow (call before saving `receipt_data`).
4. Document new configuration settings in `SETTINGS.md`.

## Export Pipeline

- **ZIP**: Contains `receipt_data.json` or `receipt_data.csv` at the root plus renamed original uploads under `receipts/`.
- **Filename mapping**: Export helpers rename uploads as `[store][ddmmyy]_[n][extension]`, using `UnknownStore` and the export date when OCR/user data is missing.
- **Data exports**: JSON/CSV use the renamed `file_name` with extension and omit local receipt paths; item columns/arrays are included only when item capture is enabled.

## Future Considerations

- Move session state persistence into a lightweight database when multi-user editing is required.
- Wrap OCR calls with retries + circuit breakers to harden against provider outages.
- Abstract exporter to support additional formats (Parquet, direct DB ingestion).

