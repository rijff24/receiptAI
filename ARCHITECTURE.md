# ScannerAI Architecture Guide

## High-Level Flow

1. **Upload** – Users upload images/PDFs through the Streamlit sidebar (`scripts/lcf_receipt_entry_streamlit.py`).
2. **Processing Queue** – Files are queued and processed sequentially, updating Streamlit `session_state`.
3. **OCR Layer** (`scannerai/ocr/`) – Receipts are run through the selected OCR provider (Gemini Vision, GPT-4 Vision, or Tesseract + GPT-3.5).
4. **Classification Layer** (`scannerai/classifiers/`) – Optional COICOP classifier enriches the results.
5. **Settings & Secrets** (`scannerai/settings/`) – UI changes are persisted via an encrypted JSON file.
6. **UI + Editing** – Users review results, adjust metadata, and export JSON/CSV.

```
Uploads -> Processing Queue -> OCR Processor -> Classification -> Session State -> UI -> Export
```

## Module Overview

| Path | Responsibility |
| ---- | -------------- |
| `scripts/lcf_receipt_entry_streamlit.py` | Streamlit UI, uploads, session state, export |
| `scannerai/ocr/` | OCR engines (`lcf_receipt_process_gemini.py`, `lcf_receipt_process_gpt4vision.py`, `lcf_receipt_process_openai.py`) |
| `scannerai/classifiers/` | COICOP classifier (model loading, inference helpers) |
| `scannerai/settings/` | `SettingsManager` (encryption, keyring, default paths) |
| `scannerai/_config/` | Legacy config reader for headless usage |
| `scannerai/utils/` | Helpers (PDF merging, token counts, etc.) |

## Settings & Secrets Flow

1. User edits values in **Application Settings**.
2. `SettingsManager` validates input, saves it under `user_settings.json`.
3. API keys are encrypted using `cryptography.Fernet`. Encryption key is stored in the OS keyring when possible.
4. OCR processors read settings each time a file is processed, so changing keys or toggles takes effect immediately.

## Session State Model

`st.session_state` keys used in the UI:

- `results`: list of processed receipts (`{"image": ..., "receipt_data": ...}`)
- `current_index`: index of the receipt being reviewed
- `receipt_status`: status tracker for each uploaded file
- `processing_queue`: pending files
- `processing_active`: boolean gate for background work
- `file_uploader_key`: rotates to clear uploads when hitting Home

## Extending OCR Providers

1. Create `scannerai/ocr/lcf_receipt_process_newprovider.py` implementing `process_receipt`.
2. Register the provider in `scripts/lcf_receipt_entry_streamlit.py` within the OCR initialization block.
3. Add settings form inputs if the provider needs API keys or custom config.
4. Update docs: README (Features + Usage) and ARCHITECTURE.

## Extending Classification

1. Add your model + artifacts under `scannerai/classifiers/`.
2. Provide a loader/inference helper similar to `lcf_classify.py`.
3. Wire the classifier into the Streamlit workflow (call before saving `receipt_data`).
4. Document new configuration settings in `SETTINGS.md`.

## Export Pipeline

- **JSON**: Serializes each `receipt_data` entry (with optional items) plus `file_name`.
- **CSV**: Flattens base receipt fields; includes item rows only when item capture is enabled.

## Future Considerations

- Move session state persistence into a lightweight database when multi-user editing is required.
- Wrap OCR calls with retries + circuit breakers to harden against provider outages.
- Abstract exporter to support additional formats (Parquet, direct DB ingestion).

