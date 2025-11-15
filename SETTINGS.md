# ScannerAI Settings Guide

ScannerAI ships with a built-in settings manager so you can configure the application without editing Python files.

## Quick Start

1. Run `streamlit run scripts/lcf_receipt_entry_streamlit.py`.
2. Open the sidebar and expand **Application Settings**.
3. Choose your OCR provider, toggle preprocessing/debug options, and set local file paths.
4. Paste your OpenAI or Gemini API keys (they are encrypted locally).
5. Click **Save Settings**. The current OCR processor will reload automatically.

## Where Settings Are Stored

| Platform | Location |
| --- | --- |
| Windows | `%APPDATA%\ScannerAI\user_settings.json` |
| macOS | `~/Library/Application Support/ScannerAI/user_settings.json` |
| Linux | `${XDG_CONFIG_HOME:-~/.config}/ScannerAI/user_settings.json` |

The directory also contains:

- `.scannerai.key` (fallback encryption key if an OS keyring is unavailable)
- `google_credentials.json` (if you upload a service-account file via the UI)

## Security Model

- API keys are encrypted with `cryptography.Fernet`.
- When available, the encryption key is stored in the OS keyring via the `keyring` library. Otherwise it is saved as `.scannerai.key` with `0600` permissions.
- The JSON settings file never leaves your device; `.gitignore` excludes these files by default.

## Manual Editing

The settings file is JSON. Scalar values can be edited manually; API keys remain encrypted.

```json
{
  "ocr_model": 2,
  "debug_mode": false,
  "enable_preprocessing": false,
  "save_processed_image": false,
  "enable_price_count": true,
  "classifier_model_path": "C:/models/classifier.sav",
  "label_encoder_path": "C:/models/encoder.pkl",
  "tesseract_cmd_path": "C:/Program Files/Tesseract-OCR/tesseract.exe",
  "google_credentials_path": "C:/Users/alex/AppData/Roaming/ScannerAI/google_credentials.json",
  "api_keys": {
    "openai": "<encrypted>",
    "gemini": "<encrypted>",
    "google": null
  }
}
```

Avoid editing the `api_keys` block manually—use the UI or delete the encrypted value to force a re-entry.

## Resetting Settings

- In the UI, click **Reset settings to defaults**.
- Or delete `user_settings.json` and `.scannerai.key`. They will be recreated the next time you open the app.

## Troubleshooting

- **Settings form does not appear**: Ensure `cryptography` and `keyring` are installed (`pip install -r requirements.txt`).
- **OCR model still uses old credentials**: After saving, the UI resets the OCR processor. If you are running headless, restart the Streamlit process.
- **Permission errors**: Make sure your user account can read/write the settings directory listed above.

