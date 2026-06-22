"""use streamlit to create interface of receipt data entry."""

import base64
import inspect
import json
import os
import re
import threading
import time
from datetime import date, datetime
from io import BytesIO
from queue import Empty, Queue
from typing import Optional


import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from streamlit.runtime.scriptrunner import RerunException, RerunData, add_script_run_ctx

from PIL import Image

# Force hosted-mode defaults when running the public Streamlit instance.
os.environ.setdefault("SCANNERAI_HOSTED_MODE", "1")

from scannerai._config.config import config
from scannerai.settings import SettingsManager
from scannerai.utils.export_utils import (
    build_export_data,
    build_export_zip,
    missing_original_file_names,
)
from scannerai.utils.scanner_utils import merge_pdf_pages

# Configure Streamlit page
st.set_page_config(
    layout="wide",
    page_title="Living Costs and Food Survey - Receipt Data Entry",
)

OCR_MODEL_OPTIONS = {
    1: "Tesseract + GPT-3.5",
    2: "GPT-4 Vision",
    3: "Gemini Vision",
}

_LOCAL_SHUTDOWN_STATE = {"port": None, "server": None}


def is_local_launcher() -> bool:
    """Return True when running inside the packaged/launcher build."""
    return os.environ.get("SCANNERAI_LOCAL_LAUNCHER") == "1"

def parse_float(value):
    """Utility to convert mixed-format numeric strings to float."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = str(value).replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None


def format_currency_string(value):
    """Format numeric-like value to a string with two decimals, else return string."""
    parsed = parse_float(value)
    if parsed is None:
        return "" if value in (None, "") else str(value)
    return f"{parsed:.2f}"


def parse_date_string(value):
    """Convert incoming string/date to a date object if possible."""
    if value in (None, "", "null"):
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, datetime):
        return value.date()
    value_str = str(value).strip()
    if not value_str:
        return None

    # Replace common separators
    normalized = value_str.replace("\\", "/")

    date_formats = [
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%Y/%m/%d",
        "%d %b %Y",
        "%d %B %Y",
        "%d.%m.%Y",
        "%m/%d/%Y",
        "%m-%d-%Y",
    ]

    for fmt in date_formats:
        try:
            return datetime.strptime(normalized, fmt).date()
        except ValueError:
            continue
    return None


def format_date_for_storage(value):
    """Return ISO formatted date string or None."""
    parsed = parse_date_string(value)
    if parsed:
        return parsed.isoformat()
    return None


def image_array_to_base64(image_array):
    """Convert numpy image array to base64 data URL."""
    pil_img = Image.fromarray(image_array)
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def force_rerun():
    """Compatibility rerun helper for older Streamlit releases."""
    raise RerunException(RerunData())


def get_settings_manager():
    """Return (and cache) the settings manager for the current session."""
    if "settings_manager" not in st.session_state:
        try:
            st.session_state.settings_manager = SettingsManager()
        except Exception as exc:  # pragma: no cover - surface to UI
            st.sidebar.error(f"Failed to initialise settings: {exc}")
            raise
    return st.session_state.settings_manager


def render_settings_panel():
    """Render the in-app settings editor."""
    try:
        settings_manager = get_settings_manager()
    except Exception:
        return

    with st.sidebar.expander("Application Settings", expanded=False):
        hosted_mode_attr = getattr(settings_manager, "is_hosted_mode", None)
        if callable(hosted_mode_attr):
            hosted_mode = hosted_mode_attr()
        elif isinstance(hosted_mode_attr, bool):
            hosted_mode = hosted_mode_attr
        else:
            hosted_mode = config.hosted_mode
        if hosted_mode:
            st.caption(
                "Hosted mode: settings live only for this session. Export them after saving to keep a local copy."
            )
        else:
            st.caption(
                f"Settings are stored locally at: `{settings_manager.settings_path}`"
            )

        import_settings_file = st.file_uploader(
            "Import encrypted settings file",
            type=["json"],
            key="settings_import_uploader",
            help="Upload the JSON file you downloaded previously.",
        )
        import_passphrase = st.text_input(
            "Import passphrase",
            type="password",
            key="settings_import_passphrase",
            help="Enter the passphrase that was used to encrypt the exported settings.",
        )
        import_triggered = st.button(
            "Import settings",
            key="import_settings_button",
            width="stretch",
        )
        if import_triggered:
            if import_settings_file is None:
                st.error("Select a settings file to import.")
            elif not import_passphrase.strip():
                st.error("Enter the passphrase for the uploaded settings file.")
            else:
                try:
                    payload_text = import_settings_file.getvalue().decode("utf-8")
                    payload = json.loads(payload_text)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    st.error("The uploaded file is not valid JSON.")
                else:
                    try:
                        settings_manager.import_encrypted(payload, import_passphrase.strip())
                    except ValueError as exc:
                        st.error(str(exc))
                    else:
                        st.session_state["app_settings"] = settings_manager.export_settings()
                        st.session_state["ocr_processor"] = None
                        st.success("Settings imported into this session.")
                        force_rerun()
        st.divider()

        # Refresh snapshot
        snapshot = settings_manager.get_settings_snapshot()
        api_status = snapshot.get("api_keys", {})
        model_keys = list(OCR_MODEL_OPTIONS.keys())
        current_model = snapshot.get("ocr_model", 2)  # Default to GPT-4 Vision
        default_index = (
            model_keys.index(current_model) if current_model in model_keys else model_keys.index(2)
        )
        existing_google_credentials_path = snapshot.get("google_credentials_path", "")
        allow_file_uploads = not hosted_mode
        classifier_model_upload = None
        label_encoder_upload = None
        google_credentials_upload = None
        if not allow_file_uploads:
            st.info(
                "File uploads (model files or Google credentials) are disabled in hosted mode. "
                "Download your settings instead and configure file paths when running locally."
            )
        
        # Settings inputs (no form wrapper)
        ocr_model = st.selectbox(
            "Preferred OCR model",
            options=model_keys,
            format_func=lambda value: OCR_MODEL_OPTIONS.get(value, f"Model {value}"),
            index=default_index,
            help="Switch between Tesseract+GPT-3.5, GPT-4 Vision, or Gemini OCR pipelines.",
        )
        debug_mode = st.toggle(
            "Enable debug logging", value=snapshot.get("debug_mode", False)
        )
        enable_preprocessing = st.toggle(
            "Enable preprocessing", value=snapshot.get("enable_preprocessing", False)
        )
        save_processed_image = st.toggle(
            "Save processed image", value=snapshot.get("save_processed_image", False)
        )
        enable_price_count = st.toggle(
            "Enable token price counting",
            value=snapshot.get("enable_price_count", False),
        )
        enable_item_capture = st.toggle(
            "Enable item capture and editing",
            value=snapshot.get("enable_item_capture", True),
            help="When enabled, you can view, add, edit, and delete individual items from receipts.",
        )

        default_zoom = st.slider(
            "Default receipt zoom",
            min_value=0.1,
            max_value=2.0,
            value=float(snapshot.get("default_zoom", 0.5) or 0.5),
            step=0.05,
            help="Sets the initial zoom level for the receipt viewer.",
        )
        vat_rate = st.number_input(
            "Default VAT rate (%)",
            min_value=0.0,
            max_value=50.0,
            value=float(snapshot.get("vat_rate", 15.0) or 15.0),
            step=0.1,
            help="Used by the Calculate VAT button. South Africa standard VAT is 15%.",
        )

        st.divider()
        
        # Classifier model path with file uploader
        classifier_path_value = snapshot.get("classifier_model_path", "")
        classifier_model_path = st.text_input(
            "Classifier model path",
            value=classifier_path_value,
            help="Path to your trained classification model file (.sav). Used for COICOP code classification. If not provided, COICOP and confidence columns will display None. You can also upload a file below.",
        )
        if allow_file_uploads:
            classifier_model_upload = st.file_uploader(
                "Upload Classifier Model",
                type=["sav"],
                key="classifier_model_uploader",
                help="Upload a .sav classifier model file. The file will be saved to your ScannerAI settings folder.",
            )
        
        # Label encoder path with file uploader
        encoder_path_value = snapshot.get("label_encoder_path", "")
        label_encoder_path = st.text_input(
            "Label encoder path",
            value=encoder_path_value,
            help="Path to your label encoder file (.pkl). This should be generated together with the trained classifier model. If not available, COICOP classification will not work. You can also upload a file below.",
        )
        if allow_file_uploads:
            label_encoder_upload = st.file_uploader(
                "Upload Label Encoder",
                type=["pkl"],
                key="label_encoder_uploader",
                help="Upload a .pkl label encoder file. The file will be saved to your ScannerAI settings folder.",
            )
        
        # Tesseract path - only show for model 1
        if ocr_model == 1:
            tesseract_path_value = snapshot.get("tesseract_cmd_path", "")
            tesseract_cmd_path = st.text_input(
                "Tesseract executable path",
                value=tesseract_path_value,
                help="Path to the Tesseract OCR executable (tesseract.exe on Windows, tesseract on Linux/Mac). Required for Tesseract + GPT-3.5 OCR model. Example: C:/Program Files/Tesseract-OCR/tesseract.exe. Note: Tesseract must be installed on the system.",
            )
        else:
            tesseract_cmd_path = snapshot.get("tesseract_cmd_path", "")
        
        # Google credentials - only show for model 3
        if ocr_model == 3:
            google_creds_value = snapshot.get("google_credentials_path", existing_google_credentials_path)
            google_credentials_path = st.text_input(
                "Google credentials JSON path",
                value=google_creds_value,
                help="Path to your Google service account JSON file. Required for Gemini OCR. This file contains credentials for accessing Google Cloud services. You can also upload a file below.",
            )
            
            if allow_file_uploads:
                google_credentials_upload = st.file_uploader(
                    "Upload Google service account JSON",
                    type=["json"],
                    key="google_credentials_uploader",
                    help="Uploaded files are stored only on your machine inside the ScannerAI settings folder.",
                )
        else:
            google_credentials_path = existing_google_credentials_path
            google_credentials_upload = None

        st.divider()
        
        # OpenAI API key - only show for models 1 and 2
        if ocr_model in (1, 2):
            openai_has_key = api_status.get("openai", False)
            openai_key_input = st.text_input(
                "OpenAI API key",
                type="password",
                value="",
                placeholder="Enter new key" if not openai_has_key else "Key stored. Enter to replace.",
                help="Required for GPT-based OCR models. Keys are encrypted locally. Leave blank to keep the existing key.",
            )
            openai_clear = st.checkbox(
                "Remove stored OpenAI key",
                value=False,
                help="Tick to delete the stored key if you want to remove access.",
            )
        else:
            openai_key_input = ""
            openai_clear = False

        # Gemini API key - only show for model 3
        if ocr_model == 3:
            gemini_has_key = api_status.get("gemini", False)
            gemini_key_input = st.text_input(
                "Gemini API key",
                type="password",
                value="",
                placeholder="Enter new key" if not gemini_has_key else "Key stored. Enter to replace.",
                help="Required for Gemini OCR. Keys are encrypted locally. Leave blank to keep the existing key.",
            )
            gemini_clear = st.checkbox(
                "Remove stored Gemini key",
                value=False,
                help="Tick to delete the stored Gemini key.",
            )
        else:
            gemini_key_input = ""
            gemini_clear = False

        if hosted_mode:
            export_passphrase = st.text_input(
                "Passphrase for exported settings",
                type="password",
                key="settings_export_passphrase",
                help="Required to encrypt the settings file that you can download after saving.",
            )
        else:
            export_passphrase = st.text_input(
                "Passphrase for exported settings (optional)",
                type="password",
                key="settings_export_passphrase",
                help="Provide a passphrase if you want to download an encrypted backup after saving.",
            )

        save_settings = st.button("Save Settings", width="stretch")

        if save_settings:
            validation_errors = []
            fallback_openai_available = bool(config.openai_api_key) or bool(
                config.openai_api_key_path and os.path.exists(config.openai_api_key_path)
            )
            fallback_gemini_available = bool(config.gemini_api_key) or bool(
                config.gemini_api_key_path and os.path.exists(config.gemini_api_key_path)
            )
            fallback_google_credentials_available = bool(
                config.google_credentials_path and os.path.exists(config.google_credentials_path)
            )
            if hosted_mode and not export_passphrase.strip():
                validation_errors.append(
                    "Enter a passphrase so your settings can be encrypted for download."
                )
            if ocr_model in (1, 2) and not (
                openai_key_input.strip()
                or api_status.get("openai")
                or fallback_openai_available
            ):
                validation_errors.append(
                    "An OpenAI API key is required for GPT-based OCR models."
                )
            if ocr_model == 3:
                if not (
                    gemini_key_input.strip()
                    or api_status.get("gemini")
                    or fallback_gemini_available
                ):
                    validation_errors.append("A Gemini API key is required for Gemini OCR.")
                if not (
                    google_credentials_upload
                    or google_credentials_path.strip()
                    or existing_google_credentials_path.strip()
                    or fallback_google_credentials_available
                ):
                    validation_errors.append(
                        "Google service-account credentials are required for Gemini OCR."
                    )

            if validation_errors:
                for message in validation_errors:
                    st.error(message)
                st.info("Update the missing fields above and press Save Settings again.")
                return

            updates = {
                "ocr_model": ocr_model,
                "debug_mode": debug_mode,
                "enable_preprocessing": enable_preprocessing,
                "save_processed_image": save_processed_image,
                "enable_price_count": enable_price_count,
                "enable_item_capture": enable_item_capture,
                "default_zoom": float(default_zoom),
                "vat_rate": float(vat_rate),
                "classifier_model_path": classifier_model_path.strip(),
                "label_encoder_path": label_encoder_path.strip(),
                "tesseract_cmd_path": tesseract_cmd_path.strip(),
                "google_credentials_path": google_credentials_path.strip(),
            }

            # Handle uploaded classifier model
            if classifier_model_upload is not None:
                classifier_path = settings_manager.settings_dir / "classifier_model.sav"
                classifier_path.write_bytes(classifier_model_upload.getvalue())
                settings_manager.secure_file(classifier_path)
                updates["classifier_model_path"] = str(classifier_path)

            # Handle uploaded label encoder
            if label_encoder_upload is not None:
                encoder_path = settings_manager.settings_dir / "label_encoder.pkl"
                encoder_path.write_bytes(label_encoder_upload.getvalue())
                settings_manager.secure_file(encoder_path)
                updates["label_encoder_path"] = str(encoder_path)

            # Handle uploaded Google credentials
            if google_credentials_upload is not None:
                credentials_path = settings_manager.settings_dir / "google_credentials.json"
                credentials_path.write_bytes(google_credentials_upload.getvalue())
                settings_manager.secure_file(credentials_path)
                updates["google_credentials_path"] = str(credentials_path)

            settings_manager.update_values(updates)

            if openai_key_input.strip():
                settings_manager.set_api_key("openai", openai_key_input.strip())
            elif openai_clear:
                settings_manager.set_api_key("openai", None)

            if gemini_key_input.strip():
                settings_manager.set_api_key("gemini", gemini_key_input.strip())
            elif gemini_clear:
                settings_manager.set_api_key("gemini", None)

            st.session_state["app_settings"] = settings_manager.export_settings()

            export_passphrase_clean = export_passphrase.strip()
            if export_passphrase_clean:
                try:
                    export_payload = settings_manager.export_encrypted(export_passphrase_clean)
                except ValueError as exc:
                    st.error(str(exc))
                    return
                st.session_state["settings_export_blob"] = json.dumps(export_payload, indent=2)
            else:
                st.session_state.pop("settings_export_blob", None)

            st.session_state["ocr_processor"] = None
            if hosted_mode:
                st.success(
                    "Settings saved for this session. Download the encrypted file below to reuse them later."
                )
            else:
                st.success("Settings saved locally. Restart processing to apply changes.")

        export_blob = st.session_state.get("settings_export_blob")
        if export_blob:
            st.download_button(
                "Download encrypted settings file",
                data=export_blob,
                file_name=SettingsManager.EXPORT_FILENAME,
                mime="application/json",
                width="stretch",
            )

        reset_clicked = st.button(
            "Reset settings to defaults",
            type="secondary",
            width="stretch",
        )
        if reset_clicked:
            defaults = settings_manager.get_defaults()
            scalar_defaults = {key: value for key, value in defaults.items() if key != "api_keys"}
            settings_manager.update_values(scalar_defaults)
            for provider in ("openai", "gemini", "google"):
                settings_manager.set_api_key(provider, None)
            st.session_state["app_settings"] = settings_manager.export_settings()
            st.session_state.pop("settings_export_blob", None)
            st.session_state["ocr_processor"] = None
            if hosted_mode:
                st.success("Settings restored to defaults for this session. Save and download the file to keep them.")
            else:
                st.success("Settings restored to defaults.")


def process_image(image_path, ocr_processor, app_settings=None):
    """Process a single receipt image."""
    if app_settings is None:
        app_settings = st.session_state.get("app_settings", {})
    
    # Read the image
    if image_path.lower().endswith((".png", ".jpg", ".jpeg")):
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    elif image_path.lower().endswith(".pdf"):
        original_image = merge_pdf_pages(image_path)
        original_image = np.array(original_image)
    
    receipt_data = {
        "shop_name": None,
        "payment_mode": None,
        "total_amount": None,
        "vat_amount": 0,
        "transaction_date": None,
        "notes": "",
        "items": [],
        "receipt_pathfile": image_path,
    }
    
    # Process receipt using OCR
    if ocr_processor:
        process_kwargs = {}
        try:
            signature = inspect.signature(ocr_processor.process_receipt)
            if "enable_price_count" in signature.parameters:
                process_kwargs["enable_price_count"] = app_settings.get(
                    "enable_price_count", False
                )
            if "debug_mode" in signature.parameters:
                process_kwargs["debug_mode"] = app_settings.get("debug_mode", False)
        except (AttributeError, ValueError, TypeError):  # pragma: no cover
            process_kwargs = {}

        processed_data = ocr_processor.process_receipt(image_path, **process_kwargs)
        if isinstance(processed_data, dict):
            receipt_data.update(processed_data)

    receipt_data["transaction_date"] = format_date_for_storage(
        receipt_data.get("transaction_date")
    )
    
    # Normalise receipt structure
    receipt_data.setdefault("shop_name", None)
    receipt_data.setdefault("payment_mode", None)
    receipt_data.setdefault("total_amount", None)
    receipt_data.setdefault("vat_amount", 0)
    receipt_data.setdefault("transaction_date", None)
    receipt_data.setdefault("notes", "")
    
    # Preserve items from OCR if they exist, otherwise initialize empty list
    if "items" not in receipt_data or not receipt_data["items"]:
        receipt_data["items"] = []
    elif isinstance(receipt_data["items"], list):
        # Normalize item structure to ensure consistent field names
        normalized_items = []
        for item in receipt_data["items"]:
            if isinstance(item, dict):
                normalized_item = {
                    "item_name": item.get("item_name") or item.get("name", ""),
                    "price": item.get("price"),
                    "coicop": item.get("coicop") or item.get("code"),
                    "coicop_desc": item.get("coicop_desc") or item.get("code_desc"),
                    "confidence": item.get("confidence") or item.get("prob"),
                }
                normalized_items.append(normalized_item)
        receipt_data["items"] = normalized_items

    return {"image": original_image, "receipt_data": receipt_data}


def process_file_bytes(file_name, file_bytes, ocr_processor, app_settings=None):
    """Process an in-memory file by writing it to a temporary location."""
    temp_suffix = abs(hash((file_name, time.time())))
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", file_name)
    temp_path = f"temp_{temp_suffix}_{safe_name}"
    with open(temp_path, "wb") as temp_file:
        temp_file.write(file_bytes)
    try:
        return process_image(temp_path, ocr_processor, app_settings=app_settings)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def attach_original_upload(result, file_name, file_bytes):
    """Attach original upload metadata needed for renamed ZIP exports."""
    if not result:
        return result
    if isinstance(file_bytes, bytes):
        payload = file_bytes
    elif isinstance(file_bytes, bytearray):
        payload = bytes(file_bytes)
    else:
        payload = b""

    result["file_name"] = file_name
    result["original_file_name"] = file_name
    result["original_file_bytes"] = payload
    return result


def save_to_json(results, file_path):
    """Save results to JSON file."""
    serializable_results = []
    for result in results:
        receipt_data = {
            **result["receipt_data"],
            "transaction_date": format_date_for_storage(
                result["receipt_data"].get("transaction_date")
            ),
            "notes": result["receipt_data"].get("notes", ""),
        }
        receipt_data.pop("receipt_pathfile", None)
        serializable_results.append(
            {
                "file_name": result.get("file_name", ""),
                "receipt_data": receipt_data,
            }
        )
    with open(file_path, "w") as json_file:
        json.dump(serializable_results, json_file, indent=4)


def save_to_csv(results, file_path):
    """Save results to CSV file."""
    rows = []
    for result in results:
        receipt_data = result["receipt_data"]
        file_name = result.get("file_name") or os.path.basename(
            receipt_data.get("receipt_pathfile", "")
        )
        if not receipt_data["items"]:
            rows.append(
                {
                    "file_name": file_name,
                    "item": "",
                    "code": "",
                    "code_desc": "",
                    "price": "",
                    "prob": "",
                    "shop_name": receipt_data["shop_name"],
                    "payment_mode": receipt_data.get("payment_mode", ""),
                    "total_amount": receipt_data.get("total_amount", ""),
                    "vat_amount": receipt_data.get("vat_amount", ""),
                    "transaction_date": receipt_data.get("transaction_date", ""),
                    "notes": receipt_data.get("notes", ""),
                }
            )
        else:
            for item in receipt_data["items"]:
                rows.append(
                    {
                        "file_name": file_name,
                        "item": item.get("name", ""),
                        "code": item.get("code", ""),
                        "code_desc": item.get("code_desc", ""),
                        "price": item.get("price", ""),
                        "prob": item.get("prob", ""),
                        "shop_name": receipt_data["shop_name"],
                        "payment_mode": receipt_data.get("payment_mode", ""),
                        "total_amount": receipt_data.get("total_amount", ""),
                        "vat_amount": receipt_data.get("vat_amount", ""),
                        "transaction_date": receipt_data.get("transaction_date", ""),
                        "notes": receipt_data.get("notes", ""),
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(file_path, index=False)


def autosave_results():
    """Persist current results after each receipt is handled."""
    autosave_path = st.session_state.get("autosave_path", "receipt_autosave.json")
    if st.session_state.results:
        try:
            save_to_json(st.session_state.results, autosave_path)
        except OSError as exc:  # pragma: no cover
            st.warning(f"Autosave failed: {exc}")


def ensure_receipt_widget_uid(result: dict) -> str:
    """Return a stable per-result widget UID for receipt form keys."""
    uid = result.get("_widget_uid")
    if not uid:
        counter = st.session_state.get("receipt_widget_uid_counter", 0) + 1
        st.session_state.receipt_widget_uid_counter = counter
        uid = f"receipt_{counter}"
        result["_widget_uid"] = uid
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(uid))


def receipt_widget_key(result: dict, field_name: str) -> str:
    """Build a Streamlit key scoped to a specific receipt result."""
    return f"{field_name}_{ensure_receipt_widget_uid(result)}"


def clear_receipt_form_widget_state() -> None:
    """Clear Streamlit receipt form widget state so deleted data cannot leak."""
    prefixes = (
        "shop_name_", "total_amount_", "vat_amount_", "transaction_date_",
        "payment_mode_select_", "payment_mode_manual_", "notes_",
        "vat_calc_notice_", "new_item_name_", "new_item_price_", "new_item_coicop_",
        "new_item_coicop_desc_", "new_item_confidence_", "add_item_", "calculate_vat_",
    )
    to_remove = []
    for key in list(st.session_state.keys()):
        if key.startswith(prefixes):
            to_remove.append(key)
            continue
        if key.startswith("delete_item_"):
            to_remove.append(key)
    for k in to_remove:
        st.session_state.pop(k, None)


def delete_receipt_at(index: int) -> None:
    """Remove a receipt from results and update related state."""
    results = st.session_state.get("results", [])
    if not results or index < 0 or index >= len(results):
        return

    removed = results.pop(index)
    receipt_data = removed.get("receipt_data", {}) or {}
    pathfile = receipt_data.get("receipt_pathfile")
    file_name = removed.get("file_name")

    identifiers = set()
    if file_name:
        identifiers.add(file_name)
    if pathfile:
        identifiers.add(pathfile)
        identifiers.add(os.path.basename(pathfile))

    display_name = file_name or (os.path.basename(pathfile) if pathfile else f"Receipt {index + 1}")

    # Remove matching status entries
    status_entries = st.session_state.get("receipt_status", [])
    st.session_state.receipt_status = [
        entry for entry in status_entries if entry.get("file") not in identifiers
    ]

    # Update process counts
    counts = st.session_state.get("process_counts", {"completed": 0, "total": 0})
    counts["total"] = max(counts.get("total", 0) - 1, 0)
    counts["completed"] = min(counts.get("completed", 0), counts["total"])
    st.session_state.process_counts = counts

    # Adjust current index
    remaining = len(results)
    if remaining == 0:
        st.session_state.current_index = 0
    elif st.session_state.current_index >= remaining:
        st.session_state.current_index = remaining - 1

    clear_receipt_form_widget_state()
    autosave_results()
    st.session_state["receipt_deleted_notice"] = f"Deleted {display_name}."


def handle_calculate_vat(
    receipt_index: int,
    vat_state_key: str,
    notice_state_key: Optional[str] = None,
) -> None:
    """Callback to back-calculate VAT for a receipt."""
    results = st.session_state.get("results", [])
    if not results or receipt_index < 0 or receipt_index >= len(results):
        return

    receipt_data = results[receipt_index]["receipt_data"]
    total_amount_value = parse_float(receipt_data.get("total_amount"))
    app_settings = st.session_state.get("app_settings", {})
    configured_rate = app_settings.get("vat_rate", 15.0)

    notice_key = notice_state_key or f"vat_calc_notice_{receipt_index}"

    try:
        configured_rate = float(configured_rate)
    except (TypeError, ValueError):
        configured_rate = 15.0

    if total_amount_value is None or total_amount_value <= 0:
        st.session_state[notice_key] = "Enter a valid total amount before calculating VAT."
        return
    if configured_rate <= 0:
        st.session_state[notice_key] = "VAT rate must be greater than zero."
        return

    st.session_state.pop(notice_key, None)
    vat_fraction = configured_rate / (100.0 + configured_rate)
    computed_vat = round(total_amount_value * vat_fraction, 2)

    st.session_state[vat_state_key] = computed_vat
    receipt_data["vat_amount"] = computed_vat


def sync_current_receipt_form_state() -> None:
    """Sync current receipt form widgets into result data before export."""
    results = st.session_state.get("results", [])
    current_index = st.session_state.get("current_index", 0)
    if not results or current_index < 0 or current_index >= len(results):
        return

    current_result = results[current_index]
    receipt_data = current_result.setdefault("receipt_data", {})

    shop_key = receipt_widget_key(current_result, "shop_name")
    total_key = receipt_widget_key(current_result, "total_amount")
    vat_key = receipt_widget_key(current_result, "vat_amount")
    date_key = receipt_widget_key(current_result, "transaction_date")
    payment_select_key = receipt_widget_key(current_result, "payment_mode_select")
    payment_manual_key = receipt_widget_key(current_result, "payment_mode_manual")
    notes_key = receipt_widget_key(current_result, "notes")

    if shop_key in st.session_state:
        receipt_data["shop_name"] = st.session_state.get(shop_key)

    if total_key in st.session_state:
        total_value = parse_float(st.session_state.get(total_key))
        receipt_data["total_amount"] = (
            f"{total_value:.2f}" if total_value is not None else None
        )

    if vat_key in st.session_state:
        vat_value = parse_float(st.session_state.get(vat_key))
        receipt_data["vat_amount"] = round(vat_value, 2) if vat_value is not None else 0

    if date_key in st.session_state:
        date_value = str(st.session_state.get(date_key) or "").strip()
        formatted_date = format_date_for_storage(date_value)
        receipt_data["transaction_date"] = (
            formatted_date if formatted_date else date_value or None
        )

    if payment_select_key in st.session_state:
        selected_payment = st.session_state.get(payment_select_key)
        if selected_payment == "Enter manually":
            receipt_data["payment_mode"] = str(
                st.session_state.get(payment_manual_key) or ""
            ).strip()
        else:
            receipt_data["payment_mode"] = selected_payment

    if notes_key in st.session_state:
        receipt_data["notes"] = str(st.session_state.get(notes_key) or "").strip()


def navigate_receipt(delta: int) -> None:
    """Move to another receipt while preserving visible form edits."""
    results = st.session_state.get("results", [])
    if not results:
        return

    current_index = st.session_state.get("current_index", 0)
    target_index = min(max(current_index + delta, 0), len(results) - 1)
    if target_index == current_index:
        return

    sync_current_receipt_form_state()
    st.session_state.current_index = target_index
    autosave_results()


def process_receipts_worker(files_data, ocr_processor, app_settings, event_queue, cancel_event):
    """Background worker to process receipts sequentially."""
    for entry in files_data:
        if cancel_event.is_set():
            break

        file_name = entry.get("name", "Receipt")
        event_queue.put({"event": "status", "file": file_name, "status": "processing"})
        try:
            file_bytes = entry.get("data", b"")
            result = process_file_bytes(
                file_name,
                file_bytes,
                ocr_processor,
                app_settings=app_settings,
            )
        except Exception as exc:  # pragma: no cover
            event_queue.put(
                {
                    "event": "error",
                    "file": file_name,
                    "error": str(exc),
                    "data": entry.get("data", b""),
                }
            )
        else:
            if result:
                attach_original_upload(result, file_name, file_bytes)
                if "receipt_data" in result:
                    result["receipt_data"]["processing_status"] = "processed"
                event_queue.put({"event": "result", "file": file_name, "result": result})
            else:
                event_queue.put(
                    {
                        "event": "error",
                        "file": file_name,
                        "error": "No data returned.",
                        "data": entry.get("data", b""),
                    }
                )

    if cancel_event.is_set():
        event_queue.put({"event": "cancelled"})
    event_queue.put({"event": "done"})


def start_processing_thread(files_data):
    """Spawn the processing worker thread and supporting structures."""
    if not files_data:
        return

    existing_thread = st.session_state.get("processing_thread")
    if existing_thread and existing_thread.is_alive():
        return

    event_queue = Queue()
    cancel_event = threading.Event()

    app_settings = json.loads(json.dumps(st.session_state.get("app_settings", {})))
    ocr_processor = st.session_state.get("ocr_processor")

    worker = threading.Thread(
        target=process_receipts_worker,
        args=(files_data, ocr_processor, app_settings, event_queue, cancel_event),
        daemon=True,
    )
    add_script_run_ctx(worker)
    worker.start()

    st.session_state.processing_thread = worker
    st.session_state.processing_event_queue = event_queue
    st.session_state.processing_cancel_event = cancel_event


def drain_processing_events():
    """Apply pending processing events emitted by the worker."""
    event_summary = {
        "events_applied": False,
        "result_added": False,
        "completed": False,
        "cancelled": False,
    }
    event_queue = st.session_state.get("processing_event_queue")
    if not event_queue:
        return event_summary

    events_applied = False
    while True:
        try:
            event = event_queue.get_nowait()
        except Empty:
            break

        events_applied = True
        event_summary["events_applied"] = True
        event_type = event.get("event")
        file_name = event.get("file")

        if event_type == "status" and file_name:
            update_receipt_status(file_name, event.get("status", "processing"))

        elif event_type == "result" and file_name:
            result = event.get("result")
            if result:
                st.session_state.results.append(result)
                event_summary["result_added"] = True
                update_receipt_status(file_name, "processed")
                st.session_state.processing_payloads.pop(file_name, None)
                queue_list = st.session_state.get("processing_queue", [])
                if file_name in queue_list:
                    queue_list.remove(file_name)
                counts = st.session_state.get("process_counts", {"completed": 0, "total": 0})
                counts["completed"] = min(counts.get("completed", 0) + 1, counts.get("total", 0))
                st.session_state.process_counts = counts
                autosave_results()

        elif event_type == "error" and file_name:
            error_message = event.get("error", "Processing failed.")
            update_receipt_status(file_name, "error", error_message)
            st.session_state.processing_payloads.pop(file_name, None)
            queue_list = st.session_state.get("processing_queue", [])
            if file_name in queue_list:
                queue_list.remove(file_name)
            st.session_state.failed_receipts.append(
                {
                    "name": file_name,
                    "data": event.get("data"),
                    "error": error_message,
                }
            )
            counts = st.session_state.get("process_counts", {"completed": 0, "total": 0})
            counts["completed"] = min(counts.get("completed", 0) + 1, counts.get("total", 0))
            st.session_state.process_counts = counts

        elif event_type == "cancelled":
            event_summary["cancelled"] = True
            st.session_state["processing_cancelled_notice"] = "Processing cancelled. Partial results are available below."

        elif event_type == "done":
            event_summary["completed"] = True
            st.session_state.processing_active = False
            st.session_state.processing_thread = None
            st.session_state.processing_cancel_event = None
            st.session_state.processing_event_queue = None
            if not st.session_state.get("processing_queue"):
                st.session_state.processing_payloads = {}
            if not st.session_state.get("processing_queue"):
                st.session_state["processing_completed_notice"] = "Processing complete."

    if events_applied:
        update_process_counts()

    return event_summary


def set_processing_status_page(page: int) -> None:
    """Set the processing status pagination page."""
    st.session_state["processing_status_page"] = max(page, 0)


def ordered_processing_status_entries(status_entries):
    """Return statuses with the active receipt first."""
    current_processing_file = next(
        (entry["file"] for entry in status_entries if entry.get("status") == "processing"),
        None,
    )
    if not current_processing_file:
        return list(status_entries)
    return (
        [entry for entry in status_entries if entry["file"] == current_processing_file]
        + [entry for entry in status_entries if entry["file"] != current_processing_file]
    )


def render_processing_status_entries(status_entries):
    """Render paged receipt processing detail rows."""
    if not status_entries:
        return

    ordered = ordered_processing_status_entries(status_entries)
    if ordered and ordered[0].get("status") == "processing":
        st.session_state["processing_status_page"] = 0

    per_page = 8
    total = len(ordered)
    num_pages = max(1, (total + per_page - 1) // per_page)
    page_key = "processing_status_page"
    page = min(st.session_state.get(page_key, 0), num_pages - 1)
    st.session_state[page_key] = page
    start = page * per_page
    page_entries = ordered[start : start + per_page]

    st.write("Receipt status:")
    for entry in page_entries:
        status = entry.get("status", "unknown")
        file_name = entry.get("file", "Receipt")
        message = entry.get("message")
        label = {
            "processed": "Done",
            "skipped": "Skipped",
            "error": "Error",
            "processing": "Processing",
        }.get(status, status.capitalize())
        text = f"{label}: {file_name} - {status.capitalize()}"
        if message and status not in {"processed", "skipped"}:
            text += f" ({message})"
        st.write(text)

    if num_pages > 1:
        st.caption(f"Page {page + 1} of {num_pages} ({total} total)")
        pcols = st.columns(2)
        with pcols[0]:
            st.button(
                "Prev",
                key="processing_status_prev",
                disabled=page <= 0,
                on_click=set_processing_status_page,
                args=(page - 1,),
            )
        with pcols[1]:
            st.button(
                "Next",
                key="processing_status_next",
                disabled=page >= num_pages - 1,
                on_click=set_processing_status_page,
                args=(page + 1,),
            )


def render_processing_panel(expanded: Optional[bool] = None):
    """Render sidebar processing summary."""
    counts = st.session_state.get("process_counts", {"completed": 0, "total": 0})
    processing_active = st.session_state.get("processing_active", False)
    queue = st.session_state.get("processing_queue", [])
    receipt_status = st.session_state.get("receipt_status", [])
    current_processing = next(
        (entry["file"] for entry in receipt_status if entry.get("status") == "processing"),
        None,
    )

    has_activity = processing_active or counts.get("total") or queue or receipt_status
    if not has_activity:
        return

    if expanded is None:
        expanded = processing_active

    with st.expander("Processing status", expanded=expanded):
        total_count = counts.get("total", 0)
        total = max(total_count, 1)
        completed = counts.get("completed", 0)
        progress_ratio = min(max(completed / total, 0.0), 1.0)
        st.progress(progress_ratio, text=f"Completed {completed} / {total_count}")

        if processing_active:
            if current_processing:
                st.write(f"Currently processing **{current_processing}**")
            elif queue:
                st.write(f"Currently processing **{queue[0]}**")
            else:
                st.write("Finishing up current receipt...")

        if not processing_active and total_count:
            st.caption("Processing complete.")

        if queue:
            st.caption(f"{len(queue)} receipt(s) remaining in the queue.")

        render_processing_status_entries(receipt_status)


@st.fragment(run_every=0.5)
def render_active_processing_sidebar():
    """Refresh processing status without rerunning the review workspace."""
    had_results = bool(st.session_state.get("results"))
    was_processing = st.session_state.get("processing_active", False)
    event_summary = drain_processing_events()
    processing_active = st.session_state.get("processing_active", False)

    render_sidebar_receipt_navigation()
    if st.session_state.get("results"):
        st.divider()

    render_processing_panel(expanded=processing_active)

    if processing_active:
        if st.button(
            "Cancel Processing",
            type="secondary",
            width="stretch",
            help="Stop processing remaining receipts and keep partial results.",
        ):
            if cancel_processing():
                st.rerun()

    if event_summary["result_added"] and not had_results:
        st.rerun()

    if was_processing and not st.session_state.get("processing_active", False):
        st.rerun()


def render_sidebar_receipt_navigation() -> None:
    """Render receipt navigation at the top of the sidebar."""
    results = st.session_state.get("results", [])
    if not results:
        return

    current_index = min(st.session_state.get("current_index", 0), len(results) - 1)
    st.session_state.current_index = current_index
    col1, col2 = st.columns(2)
    with col1:
        st.button(
            "Previous",
            key="sidebar_previous_receipt",
            width="stretch",
            disabled=current_index <= 0,
            on_click=navigate_receipt,
            args=(-1,),
        )
    with col2:
        st.button(
            "Next",
            key="sidebar_next_receipt",
            width="stretch",
            disabled=current_index >= len(results) - 1,
            on_click=navigate_receipt,
            args=(1,),
        )
    st.write(f"Receipt {current_index + 1} of {len(results)}")


def update_receipt_status(file_name, status, message=None):
    """Update or append the processing status for a receipt."""
    found = False
    for entry in st.session_state.receipt_status:
        if entry["file"] == file_name:
            entry["status"] = status
            if message:
                entry["message"] = message
            elif "message" in entry:
                entry.pop("message")
            found = True
            break
    if not found:
        entry = {"file": file_name, "status": status}
        if message:
            entry["message"] = message
        st.session_state.receipt_status.append(entry)

    update_process_counts()


def update_process_counts():
    """Recompute completion counters based on receipt statuses."""
    total = st.session_state.process_counts.get("total", 0)
    completed = sum(
        1
        for entry in st.session_state.receipt_status
        if entry["status"] in {"processed", "skipped", "error"}
    )
    st.session_state.process_counts = {"completed": completed, "total": total}


def cancel_processing(reason: Optional[str] = None) -> bool:
    """Cancel any in-progress batch processing, preserving processed results."""
    if not st.session_state.get("processing_active"):
        return False
    queue = st.session_state.get("processing_queue", [])
    cancel_event = st.session_state.get("processing_cancel_event")
    if cancel_event:
        cancel_event.set()

    skipped_count = len(queue)
    skip_message = "Cancelled by user"
    for file_name in queue:
        update_receipt_status(file_name, "skipped", skip_message)

    st.session_state.processing_queue = []
    st.session_state.processing_active = False
    st.session_state.processing_thread = None
    st.session_state.processing_cancel_event = None

    summary = reason or "Processing cancelled. Partial results are available below."
    summary_with_count = (
        f"{summary} {skipped_count} pending receipt(s) were skipped."
        if skipped_count
        else summary
    )
    st.session_state["processing_cancelled_notice"] = summary_with_count
    return True


class _ShutdownRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A003
        return  # Silence default HTTP server logging.

    def _send_response(self, code: int) -> None:
        self.send_response(code)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/shutdown":
            self._send_response(204)
            trigger_local_shutdown()
        else:
            self._send_response(404)

    def do_GET(self) -> None:  # noqa: N802
        self.do_POST()


def ensure_local_shutdown_listener() -> Optional[int]:
    """Start the local shutdown listener (used by browser close hooks)."""
    if not is_local_launcher():
        return None
    if _LOCAL_SHUTDOWN_STATE["port"]:
        return _LOCAL_SHUTDOWN_STATE["port"]

    server = ThreadingHTTPServer(("127.0.0.1", 0), _ShutdownRequestHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    _LOCAL_SHUTDOWN_STATE["server"] = server
    _LOCAL_SHUTDOWN_STATE["port"] = port
    return port


def inject_browser_shutdown_hook() -> None:
    """Inject JS that notifies the launcher when the browser tab closes."""
    if not is_local_launcher():
        return
    if st.session_state.get("browser_shutdown_hook_installed"):
        return

    port = ensure_local_shutdown_listener()
    if not port:
        return

    script = f"""
    <script>
    (function() {{
        const endpoint = "http://127.0.0.1:{port}/shutdown";
        const notify = () => {{
            if (navigator.sendBeacon) {{
                navigator.sendBeacon(endpoint, "");
            }} else {{
                fetch(endpoint, {{
                    method: "POST",
                    keepalive: true
                }});
            }}
        }};
        window.addEventListener("beforeunload", notify, {{ once: true }});
    }})();
    </script>
    """
    components.html(script, height=0)
    st.session_state["browser_shutdown_hook_installed"] = True


def trigger_local_shutdown(delay: float = 0.5) -> None:
    """Schedule a local process shutdown after a short delay."""
    def _shutdown():
        time.sleep(max(delay, 0))
        os._exit(0)

    threading.Thread(target=_shutdown, daemon=True).start()


def request_exit(message: Optional[str] = None) -> None:
    """Mark the current session for exit and stop further processing."""
    st.session_state.processing_queue = []
    st.session_state.processing_active = False
    st.session_state["exit_requested"] = True
    st.session_state["exit_message"] = (
        message or "Session closed. You can safely close this tab."
    )
    st.session_state["exit_should_shutdown"] = is_local_launcher()


def initialize_session_state():
    """Initialize or reset session state variables."""
    try:
        settings_manager = get_settings_manager()
    except Exception:  # pragma: no cover - fallback if settings cannot load
        settings_manager = None

    def resolve_setting(key, fallback):
        if not settings_manager:
            return fallback
        value = settings_manager.get_value(key, fallback)
        if isinstance(value, str):
            value = value.strip()
            return value if value else fallback
        return value

    ocr_model_value = resolve_setting("ocr_model", config.ocr_model)
    try:
        ocr_model_setting = int(ocr_model_value)
    except (TypeError, ValueError):
        ocr_model_setting = config.ocr_model

    resolved_settings = {
        "ocr_model": ocr_model_setting,
        "debug_mode": resolve_setting("debug_mode", config.debug_mode),
        "enable_preprocessing": resolve_setting(
            "enable_preprocessing", config.enable_preprocessing
        ),
        "save_processed_image": resolve_setting(
            "save_processed_image", config.save_processed_image
        ),
        "enable_price_count": resolve_setting(
            "enable_price_count", config.enable_price_count
        ),
        "classifier_model_path": resolve_setting(
            "classifier_model_path", config.classifier_model_path
        ),
        "label_encoder_path": resolve_setting(
            "label_encoder_path", config.label_encoder_path
        ),
        "tesseract_cmd_path": resolve_setting(
            "tesseract_cmd_path", config.tesseract_cmd_path
        ),
        "google_credentials_path": resolve_setting(
            "google_credentials_path", config.google_credentials_path
        ),
        "default_zoom": resolve_setting("default_zoom", 0.5),
        "vat_rate": resolve_setting("vat_rate", 15.0),
    }
    st.session_state["app_settings"] = resolved_settings

    openai_api_key = settings_manager.get_api_key("openai") if settings_manager else None
    if not openai_api_key:
        openai_api_key = config.openai_api_key

    gemini_api_key = settings_manager.get_api_key("gemini") if settings_manager else None
    if not gemini_api_key:
        gemini_api_key = config.gemini_api_key

    if "results" not in st.session_state:
        print('Initialise st.session_state.results = []')
        st.session_state.results = []
    if "current_index" not in st.session_state:
        print('Initialise st.session_state.current_index = 0')
        st.session_state.current_index = 0
    if "receipt_status" not in st.session_state:
        st.session_state.receipt_status = []
    if "failed_receipts" not in st.session_state:
        st.session_state.failed_receipts = []
    if "process_counts" not in st.session_state:
        st.session_state.process_counts = {"completed": 0, "total": 0}
    if "processing_queue" not in st.session_state:
        st.session_state.processing_queue = []
    if "processing_active" not in st.session_state:
        st.session_state.processing_active = False
    if "processing_thread" not in st.session_state:
        st.session_state.processing_thread = None
    if "processing_event_queue" not in st.session_state:
        st.session_state.processing_event_queue = None
    if "processing_cancel_event" not in st.session_state:
        st.session_state.processing_cancel_event = None
    if "processing_payloads" not in st.session_state:
        st.session_state.processing_payloads = {}
    if "autosave_path" not in st.session_state:
        st.session_state.autosave_path = "receipt_autosave.json"
        
    # initialise OCR processor
    needs_ocr_initialise = (
        "ocr_processor" not in st.session_state or st.session_state.ocr_processor is None
    )
    if needs_ocr_initialise:
        st.session_state.ocr_processor = None

        if resolved_settings["ocr_model"] == 1:
            from scannerai.ocr.lcf_receipt_process_openai import LCFReceiptProcessOpenai

            st.session_state.ocr_processor = LCFReceiptProcessOpenai(
                openai_api_key_path=config.open_api_key_path,
                tesseract_cmd_path=resolved_settings["tesseract_cmd_path"],
                openai_api_key=openai_api_key,
            )
            if st.session_state.ocr_processor.get_InitSuccess():
                st.sidebar.info("Using OpenAI OCR Model")
            else:
                st.error("OCR processor initialization failed.")
            
        elif resolved_settings["ocr_model"] == 2:
            from scannerai.ocr.lcf_receipt_process_gpt4vision import LCFReceiptProcessGPT4Vision

            st.session_state.ocr_processor = LCFReceiptProcessGPT4Vision(
                openai_api_key_path=config.openai_api_key_path,
                openai_api_key=openai_api_key,
            )
            if st.session_state.ocr_processor.get_InitSuccess():
                st.sidebar.info("Using GPT-4 Vision OCR Model")
            else:
                st.error("OCR processor initialization failed.")
            
        elif resolved_settings["ocr_model"] == 3:
            from scannerai.ocr.lcf_receipt_process_gemini import LCFReceiptProcessGemini 

            st.session_state.ocr_processor = LCFReceiptProcessGemini(
                google_credentials_path=resolved_settings["google_credentials_path"],
                gemini_api_key_path=config.gemini_api_key_path,
                gemini_api_key=gemini_api_key,
            )
            if st.session_state.ocr_processor.get_InitSuccess():
                st.sidebar.info("Using Gemini OCR Model")
            else:
                st.error("OCR processor initialization failed.")
                
        else:
            st.error("WARNING: No OCR Model is set!")
        
def main():
    """To execute interface."""
    st.title("Receipt Data Entry System")

    # Initialize session state
    initialize_session_state()
    drain_processing_events()

    if st.session_state.get("exit_requested"):
        st.info(st.session_state.get("exit_message", "Session closed. You can close this tab."))
        if st.session_state.pop("exit_should_shutdown", False):
            trigger_local_shutdown()
        st.stop()

    cancel_notice = st.session_state.pop("processing_cancelled_notice", None)
    if cancel_notice:
        st.warning(cancel_notice)

    delete_notice = st.session_state.pop("receipt_deleted_notice", None)
    if delete_notice:
        st.success(delete_notice)

    completed_notice = st.session_state.pop("processing_completed_notice", None)
    if completed_notice:
        st.success(completed_notice)

    inject_browser_shutdown_hook()

    # Sidebar for file upload and navigation
    with st.sidebar:
        processing_active = st.session_state.get("processing_active", False)
        has_results = bool(st.session_state.results)
        if has_results and not processing_active:
            render_sidebar_receipt_navigation()
            st.divider()

        if processing_active:
            render_active_processing_sidebar()
        else:
            render_processing_panel()
            st.header("Upload & Navigation")

            # Use a key that can be reset to clear the uploader.
            st.markdown(
                """
                <style>
                    div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
                        transition: background-color 0.12s ease, border-color 0.12s ease,
                            box-shadow 0.12s ease, transform 0.12s ease;
                    }
                    div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"]:hover,
                    div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"]:focus-within {
                        background-color: #eef7ff;
                        border-color: #2d7ff9;
                        box-shadow: 0 0 0 2px rgba(45, 127, 249, 0.16);
                        transform: translateY(-1px);
                    }
                </style>
                """,
                unsafe_allow_html=True,
            )
            uploader_key = st.session_state.get("file_uploader_key", "file_uploader")
            uploaded_files = st.file_uploader(
                "Upload receipt images",
                type=["png", "jpg", "jpeg", "pdf"],
                accept_multiple_files=True,
                key=uploader_key,
            )

            if uploaded_files and st.button(
                "Process Uploaded Files",
                width="stretch",
                help="Start processing uploaded files",
            ):
                files_data = [
                    {"name": file.name, "data": file.getvalue()}
                    for file in uploaded_files
                ]

                st.session_state.results = []
                st.session_state.current_index = 0
                st.session_state.receipt_status = []
                st.session_state.failed_receipts = []
                st.session_state.processing_queue = [file["name"] for file in files_data]
                st.session_state.processing_payloads = {
                    file["name"]: file["data"] for file in files_data
                }
                st.session_state.process_counts = {
                    "completed": 0,
                    "total": len(files_data),
                }
                st.session_state.processing_active = True
                st.session_state.processing_event_queue = None
                st.session_state.processing_cancel_event = None
                st.session_state.processing_thread = None
                start_processing_thread(files_data)
                force_rerun()

        # Home button - show when results exist and processing is complete
        processing_active = st.session_state.get("processing_active", False)

        # Show controls again after active processing has finished.
        processing_complete = not processing_active
        if has_results and processing_complete:
            if st.button("🏠 Home", width="stretch", help="Return to home screen and clear current results"):
                st.session_state.results = []
                st.session_state.current_index = 0
                st.session_state.receipt_status = []
                st.session_state.failed_receipts = []
                st.session_state.processing_queue = []
                st.session_state.processing_active = False
                st.session_state.process_counts = {"completed": 0, "total": 0}
                # Reset file uploader by changing its key
                st.session_state.file_uploader_key = f"file_uploader_{st.session_state.get('uploader_reset_counter', 0) + 1}"
                st.session_state.uploader_reset_counter = st.session_state.get("uploader_reset_counter", 0) + 1
                st.rerun()
            st.divider()
        
        # Export options - show when results exist and processing is complete
        if has_results and processing_complete:
            st.header("Export Data")
            export_format = st.selectbox("Export format", ["CSV", "JSON"])

            # Get item capture setting
            settings_manager = get_settings_manager()
            snapshot = settings_manager.get_settings_snapshot()
            enable_item_capture = snapshot.get("enable_item_capture", True)

            sync_current_receipt_form_state()
            export_date = date.today()
            data_filename, _, file_plan = build_export_data(
                st.session_state.results,
                export_format,
                enable_item_capture,
                export_date=export_date,
            )
            missing_files = missing_original_file_names(file_plan)
            if missing_files:
                st.warning(
                    "Some original upload bytes are unavailable, so those receipt "
                    "files will be missing from the ZIP: "
                    + ", ".join(missing_files)
                )

            zip_data = build_export_zip(
                st.session_state.results,
                export_format,
                enable_item_capture,
                export_date=export_date,
            )
            included_file_count = len(file_plan) - len(missing_files)
            st.caption(
                f"ZIP includes {data_filename} and {included_file_count} renamed receipt file(s)."
            )
            st.download_button(
                label=f"Download {export_format} ZIP",
                data=zip_data,
                file_name=f"receipt_export_{export_date.strftime('%Y%m%d')}.zip",
                mime="application/zip",
                width="stretch",
            )
        st.divider()
        if st.button(
            "Exit Application",
            type="secondary",
            width="stretch",
            help="End this session and close the app.",
        ):
            request_exit()
            force_rerun()

    render_settings_panel()

    # Main content area
    if st.session_state.failed_receipts:
        st.warning("Some receipts could not be processed. Retry or skip to continue.")
        for idx, failed in list(enumerate(st.session_state.failed_receipts)):
            cols = st.columns([3, 1, 1])
            with cols[0]:
                error_msg = failed.get("error", "Unknown error")
                st.write(f"**{failed['name']}** — {error_msg}")
            with cols[1]:
                if st.button("Retry", key=f"retry_failed_{idx}"):
                    try:
                        retry_settings = json.loads(json.dumps(st.session_state.get("app_settings", {})))
                        result = process_file_bytes(
                            failed["name"],
                            failed["data"],
                            st.session_state.ocr_processor,
                            app_settings=retry_settings,
                        )
                        if result:
                            attach_original_upload(
                                result,
                                failed["name"],
                                failed.get("data", b""),
                            )
                            result["receipt_data"]["processing_status"] = "processed"
                            st.session_state.results.append(result)
                            update_receipt_status(failed["name"], "processed")
                            autosave_results()
                            st.session_state.failed_receipts.pop(idx)
                            force_rerun()
                        else:
                            update_receipt_status(failed["name"], "error", "No data returned.")
                            failed["error"] = "No data returned."
                            st.error("Retry failed: no data returned.")
                    except Exception as exc:  # pragma: no cover
                        update_receipt_status(failed["name"], "error", str(exc))
                        failed["error"] = str(exc)
                        st.error(f"Retry failed: {exc}")
            with cols[2]:
                if st.button("Skip", key=f"skip_failed_{idx}"):
                    placeholder = {
                        "image": None,
                        "receipt_data": {
                            "shop_name": None,
                            "payment_mode": "EFT",
                            "total_amount": None,
                            "vat_amount": 0,
                            "transaction_date": None,
                            "notes": "",
                            "items": [],
                            "receipt_pathfile": failed["name"],
                            "processing_status": "skipped",
                        },
                        "file_name": failed["name"],
                        "original_file_name": failed["name"],
                        "original_file_bytes": failed.get("data", b""),
                    }
                    st.session_state.results.append(placeholder)
                    update_receipt_status(failed["name"], "skipped", "Marked as skipped by user.")
                    autosave_results()
                    st.session_state.failed_receipts.pop(idx)
                    st.success(f"Skipped {failed['name']}")
                    force_rerun()

    if st.session_state.results:
        current_result = st.session_state.results[st.session_state.current_index]
        current_index = st.session_state.current_index

        # Display receipt image and data side by side
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Receipt Image")
            image = current_result["image"]
            if image is None:
                st.info("No image available for this receipt.")
            else:
                image_src = image_array_to_base64(image)
                viewer_id = f"receipt-viewer-{current_index}"
                image_id = f"receipt-img-{current_index}"
                zoom_in_id = f"zoom-in-{current_index}"
                zoom_out_id = f"zoom-out-{current_index}"
                reset_id = f"zoom-reset-{current_index}"
                zoom_info_id = f"zoom-info-{current_index}"
                app_settings = st.session_state.get("app_settings", {})
                try:
                    initial_zoom = float(app_settings.get("default_zoom", 0.5))
                except (TypeError, ValueError):
                    initial_zoom = 0.5
                initial_zoom = max(0.1, min(initial_zoom, 2.0))
                initial_zoom_percent = int(round(initial_zoom * 100))

                viewer_html = f"""
                <style>
                    #{viewer_id} {{
                        width: 100%;
                        height: 600px;
                        min-height: 300px;
                        max-height: none;
                        overflow: auto;
                        resize: vertical;
                        border: 1px solid #d9d9d9;
                        border-radius: 6px;
                        background: #f7f7f7;
                        position: relative;
                        cursor: grab;
                        touch-action: none;
                    }}
                    #{viewer_id} img {{
                        display: block;
                        height: auto;
                        max-width: none;
                        transform-origin: 0 0;
                        user-select: none;
                        -webkit-user-drag: none;
                        transition: transform 0.08s ease-out, width 0.08s ease-out;
                    }}
                    .zoom-controls {{
                        position: absolute;
                        top: 10px;
                        right: 10px;
                        display: flex;
                        flex-direction: column;
                        gap: 6px;
                        z-index: 5;
                    }}
                    .zoom-controls button {{
                        width: 34px;
                        height: 34px;
                        border: 1px solid #cfcfcf;
                        border-radius: 4px;
                        background: #fff;
                        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
                        font-size: 18px;
                        cursor: pointer;
                    }}
                    .zoom-controls button:hover {{
                        background: #f0f0f0;
                    }}
                    #{zoom_info_id} {{
                        position: absolute;
                        left: 10px;
                        bottom: 10px;
                        background: rgba(0, 0, 0, 0.65);
                        color: #fff;
                        font-size: 12px;
                        padding: 4px 8px;
                        border-radius: 4px;
                        z-index: 5;
                    }}
                </style>
                <div id="{viewer_id}">
                    <img id="{image_id}" src="{image_src}" alt="Receipt image" draggable="false" />
                    <div class="zoom-controls">
                        <button id="{zoom_in_id}" aria-label="Zoom in">+</button>
                        <button id="{zoom_out_id}" aria-label="Zoom out">−</button>
                        <button id="{reset_id}" aria-label="Reset zoom">⌂</button>
                    </div>
                    <div id="{zoom_info_id}">{initial_zoom_percent}%</div>
                </div>
                <script>
                    (function() {{
                        const viewer = document.getElementById("{viewer_id}");
                        const img = document.getElementById("{image_id}");
                        const zoomInBtn = document.getElementById("{zoom_in_id}");
                        const zoomOutBtn = document.getElementById("{zoom_out_id}");
                        const resetBtn = document.getElementById("{reset_id}");
                        const info = document.getElementById("{zoom_info_id}");
                        if (!viewer || !img) return;

                        const MIN_SCALE = 0.2;
                        const MAX_SCALE = 5.0;
                        const INITIAL_SCALE = {initial_zoom};
                        let scale = INITIAL_SCALE;
                        let offsetX = 0;
                        let offsetY = 0;
                        let baseImageWidth = 0;
                        let isPanning = false;
                        let startX = 0;
                        let startY = 0;
                        let startOffsetX = 0;
                        let startOffsetY = 0;
                        const activeTouches = new Map();
                        let pinchStartDistance = null;
                        let pinchStartScale = null;
                        const STORAGE_KEY = "receipt-viewer-state::{viewer_id}";
                        let pendingSaveFrame = null;

                        const clamp = (value, min, max) => Math.min(Math.max(value, min), max);
                        const syncImageBaseSize = () => {{
                            const rect = viewer.getBoundingClientRect();
                            const availableWidth = Math.max(rect.width - 24, 240);
                            baseImageWidth = availableWidth;
                        }};
                        const getDisplayWidth = () => {{
                            syncImageBaseSize();
                            return baseImageWidth * scale;
                        }};
                        const computePreferredViewerHeight = () => {{
                            const MIN_HEIGHT = 720;
                            const MAX_HEIGHT = 1700;
                            const FALLBACK = 1000;
                            let parentViewport = null;
                            try {{
                                if (window.parent && window.parent !== window) {{
                                    parentViewport = window.parent.innerHeight || null;
                                }}
                            }} catch (err) {{
                                parentViewport = null;
                            }}
                            const base = parentViewport || window.innerHeight || FALLBACK;
                            const adjusted = base ? base - 120 : FALLBACK;
                            return clamp(Math.round(adjusted), MIN_HEIGHT, MAX_HEIGHT);
                        }};

                        const applyDefaultViewerHeight = () => {{
                            if (viewer.dataset.defaultHeightApplied === "1") {{
                                return;
                            }}
                            const desired = computePreferredViewerHeight();
                            if (viewer.offsetHeight < desired) {{
                                viewer.style.height = `${{desired}}px`;
                            }}
                            viewer.dataset.defaultHeightApplied = "1";
                        }};
                        applyDefaultViewerHeight();

                        const updateFrameHeight = () => {{
                            const extra = 200;
                            const target = Math.max(
                                document.documentElement.scrollHeight,
                                viewer.offsetHeight + extra
                            );
                            const frame = window.frameElement;
                            if (frame) {{
                                frame.style.height = `${{target}}px`;
                                const block = frame.closest('[data-testid="stVerticalBlock"]');
                                if (block) {{
                                    block.style.height = "auto";
                                }}
                                const parent = frame.parentElement;
                                if (parent) {{
                                    parent.style.height = "auto";
                                }}
                            }}
                            if (window.parent && window.parent !== window) {{
                                window.parent.postMessage({{ type: "streamlit:setFrameHeight", height: target }}, "*");
                            }}
                            if (window.Streamlit && Streamlit.setFrameHeight) {{
                                Streamlit.setFrameHeight(target);
                            }}
                        }};

                        const applyTransform = () => {{
                            img.style.width = `${{getDisplayWidth()}}px`;
                            img.style.height = "auto";
                            img.style.transform = `translate(${{offsetX}}px, ${{offsetY}}px)`;
                            if (info) {{
                                info.textContent = `${{Math.round(scale * 100)}}%`;
                            }}
                            scheduleStateSave();
                            updateFrameHeight();
                        }};

                        const setScale = (newScale, originX, originY) => {{
                            const clampedScale = clamp(newScale, MIN_SCALE, MAX_SCALE);
                            const rect = viewer.getBoundingClientRect();
                            const relativeX = originX - rect.left;
                            const relativeY = originY - rect.top;
                            const imageX = (relativeX - offsetX) / scale;
                            const imageY = (relativeY - offsetY) / scale;
                            scale = clampedScale;
                            offsetX = relativeX - imageX * scale;
                            offsetY = relativeY - imageY * scale;
                            applyTransform();
                        }};

                        const centerImageHorizontalOnly = () => {{
                            const rect = viewer.getBoundingClientRect();
                            const viewportWidth = viewer.clientWidth || rect.width;
                            const displayWidth = getDisplayWidth();
                            offsetX = Math.max((viewportWidth - displayWidth) / 2, 0);
                            offsetY = 0;
                            applyTransform();
                        }};

                        const loadViewerPreferences = () => {{
                            try {{
                                const stored = sessionStorage.getItem(STORAGE_KEY);
                                if (!stored) return false;
                                const parsed = JSON.parse(stored);
                                if (parsed && parsed.height) {{
                                    viewer.style.height = `${{parsed.height}}px`;
                                }}
                                if (parsed && typeof parsed.scale === "number") {{
                                    scale = clamp(parsed.scale, MIN_SCALE, MAX_SCALE);
                                }}
                                return true;
                            }} catch (err) {{
                                console.warn("Failed to load viewer state", err);
                            }}
                            return false;
                        }};

                        const saveViewerPreferences = () => {{
                            try {{
                                const payload = JSON.stringify({{
                                    height: viewer.offsetHeight,
                                    scale,
                                }});
                                sessionStorage.setItem(STORAGE_KEY, payload);
                            }} catch (err) {{
                                console.warn("Failed to save viewer state", err);
                            }}
                        }};

                        const scheduleStateSave = () => {{
                            if (pendingSaveFrame) {{
                                return;
                            }}
                            pendingSaveFrame = requestAnimationFrame(() => {{
                                pendingSaveFrame = null;
                                saveViewerPreferences();
                            }});
                        }};

                        const startPan = (clientX, clientY) => {{
                            isPanning = true;
                            startX = clientX;
                            startY = clientY;
                            startOffsetX = offsetX;
                            startOffsetY = offsetY;
                            viewer.style.cursor = "grabbing";
                        }};

                        const movePan = (clientX, clientY) => {{
                            if (!isPanning) return;
                            offsetX = startOffsetX + (clientX - startX);
                            offsetY = startOffsetY + (clientY - startY);
                            applyTransform();
                        }};

                        const endPan = () => {{
                            if (!isPanning) return;
                            isPanning = false;
                            viewer.style.cursor = "grab";
                        }};

                        const getViewerCenter = () => {{
                            const rect = viewer.getBoundingClientRect();
                            return {{
                                x: rect.left + rect.width / 2,
                                y: rect.top + rect.height / 2,
                            }};
                        }};

                        const handleWheel = (event) => {{
                            if (event.ctrlKey) {{
                                event.preventDefault();
                                const zoomFactor = event.deltaY > 0 ? 0.9 : 1.1;
                                setScale(scale * zoomFactor, event.clientX, event.clientY);
                            }} else {{
                                offsetX -= event.deltaX;
                                offsetY -= event.deltaY;
                                applyTransform();
                                event.preventDefault();
                            }}
                        }};

                        const pointerIsTouch = (event) =>
                            event.pointerType === "touch" || event.pointerType === "pen";

                        const updateTouch = (event) => {{
                            activeTouches.set(event.pointerId, {{ x: event.clientX, y: event.clientY }});
                        }};

                        const getTouchInfo = () => {{
                            if (activeTouches.size < 2) return null;
                            const points = Array.from(activeTouches.values());
                            const dx = points[0].x - points[1].x;
                            const dy = points[0].y - points[1].y;
                            return {{
                                distance: Math.hypot(dx, dy),
                                center: {{
                                    x: (points[0].x + points[1].x) / 2,
                                    y: (points[0].y + points[1].y) / 2,
                                }},
                            }};
                        }};

                        zoomInBtn?.addEventListener("click", (event) => {{
                            event.preventDefault();
                            const center = getViewerCenter();
                            setScale(scale * 1.2, center.x, center.y);
                        }});

                        zoomOutBtn?.addEventListener("click", (event) => {{
                            event.preventDefault();
                            const center = getViewerCenter();
                            setScale(scale / 1.2, center.x, center.y);
                        }});

                        resetBtn?.addEventListener("click", (event) => {{
                            event.preventDefault();
                            scale = INITIAL_SCALE;
                            centerImageHorizontalOnly();
                        }});

                        viewer.addEventListener("wheel", handleWheel, {{ passive: false }});

                        viewer.addEventListener("mousedown", (event) => {{
                            if (event.button !== 0) return;
                            startPan(event.clientX, event.clientY);
                        }});

                        const mouseMoveListener = (event) => {{
                            if (!isPanning) return;
                            movePan(event.clientX, event.clientY);
                        }};

                        const mouseUpListener = () => {{
                            endPan();
                        }};

                        window.addEventListener("mousemove", mouseMoveListener);
                        window.addEventListener("mouseup", mouseUpListener);
                        viewer.addEventListener("mouseleave", endPan);
                        viewer.addEventListener("mouseup", () => {{
                            requestAnimationFrame(() => {{
                                saveViewerPreferences();
                                updateFrameHeight();
                            }});
                        }});
                        viewer.addEventListener("touchend", () => {{
                            requestAnimationFrame(() => {{
                                saveViewerPreferences();
                                updateFrameHeight();
                            }});
                        }});

                        viewer.addEventListener("pointerdown", (event) => {{
                            if (!pointerIsTouch(event)) return;
                            viewer.setPointerCapture(event.pointerId);
                            updateTouch(event);
                            if (activeTouches.size === 1) {{
                                startPan(event.clientX, event.clientY);
                            }} else if (activeTouches.size === 2) {{
                                pinchStartScale = scale;
                                const info = getTouchInfo();
                                pinchStartDistance = info ? info.distance : null;
                                endPan();
                            }}
                        }});

                        viewer.addEventListener("pointermove", (event) => {{
                            if (!activeTouches.has(event.pointerId)) return;
                            updateTouch(event);
                            if (activeTouches.size === 1) {{
                                event.preventDefault();
                                movePan(event.clientX, event.clientY);
                            }} else if (activeTouches.size === 2) {{
                                event.preventDefault();
                                const info = getTouchInfo();
                                if (info && pinchStartDistance) {{
                                    const factor = info.distance / pinchStartDistance;
                                    setScale(pinchStartScale * factor, info.center.x, info.center.y);
                                }}
                            }}
                        }});

                        const releaseTouch = (event) => {{
                            if (!activeTouches.has(event.pointerId)) return;
                            activeTouches.delete(event.pointerId);
                            viewer.releasePointerCapture(event.pointerId);
                            if (activeTouches.size === 0) {{
                                pinchStartDistance = null;
                                pinchStartScale = null;
                                endPan();
                            }} else if (activeTouches.size === 1) {{
                                const remaining = Array.from(activeTouches.values())[0];
                                pinchStartDistance = null;
                                pinchStartScale = null;
                                startPan(remaining.x, remaining.y);
                            }}
                        }};

                        viewer.addEventListener("pointerup", releaseTouch);
                        viewer.addEventListener("pointercancel", releaseTouch);

                        const initialize = () => {{
                            syncImageBaseSize();
                            loadViewerPreferences();
                            centerImageHorizontalOnly();
                            updateFrameHeight();
                        }};

                        if (img.complete) {{
                            initialize();
                        }} else {{
                            img.addEventListener("load", initialize, {{ once: true }});
                        }}

                        if (window.ResizeObserver) {{
                            const ro = new ResizeObserver(() => {{
                                const previousBaseWidth = baseImageWidth;
                                syncImageBaseSize();
                                if (previousBaseWidth && Math.abs(previousBaseWidth - baseImageWidth) > 1) {{
                                    centerImageHorizontalOnly();
                                    return;
                                }}
                                saveViewerPreferences();
                                updateFrameHeight();
                            }});
                            ro.observe(viewer);
                        }} else {{
                            setInterval(updateFrameHeight, 750);
                        }}

                        return () => {{
                            window.removeEventListener("mousemove", mouseMoveListener);
                            window.removeEventListener("mouseup", mouseUpListener);
                        }};
                    }})();
                </script>
                """

                components.html(viewer_html, height=1000, width=None)

        with col2:
            st.subheader("Receipt Data")

            # Shop details
            receipt_data = current_result["receipt_data"]

            action_cols = st.columns([1, 1.4, 1])
            with action_cols[0]:
                st.button(
                    "Previous",
                    key=f"main_previous_receipt_{current_index}",
                    width="stretch",
                    disabled=current_index <= 0,
                    on_click=navigate_receipt,
                    args=(-1,),
                )
            with action_cols[1]:
                if st.button(
                    "Delete Receipt",
                    key=f"delete_receipt_{current_index}",
                    type="secondary",
                    width="stretch",
                    help="Remove this receipt from the current session.",
                ):
                    delete_receipt_at(current_index)
                    force_rerun()
            with action_cols[2]:
                st.button(
                    "Next",
                    key=f"main_next_receipt_{current_index}",
                    width="stretch",
                    disabled=current_index >= len(st.session_state.results) - 1,
                    on_click=navigate_receipt,
                    args=(1,),
                )

            # Create unique keys for each input field
            shop_key = receipt_widget_key(current_result, "shop_name")
            total_key = receipt_widget_key(current_result, "total_amount")
            vat_key = receipt_widget_key(current_result, "vat_amount")
            date_key = receipt_widget_key(current_result, "transaction_date")
            payment_select_key = receipt_widget_key(current_result, "payment_mode_select")
            payment_manual_key = receipt_widget_key(current_result, "payment_mode_manual")
            notes_key = receipt_widget_key(current_result, "notes")
            vat_notice_key = receipt_widget_key(current_result, "vat_calc_notice")

            # 1. Shop Name
            new_shop_name = st.text_input(
                "Shop Name",
                value=receipt_data["shop_name"],
                key=shop_key,
            )
            receipt_data["shop_name"] = new_shop_name

            # 2. Total Amount
            parsed_initial_total = parse_float(receipt_data.get("total_amount"))
            if parsed_initial_total is None or parsed_initial_total < 0:
                parsed_initial_total = 0.0
            # Seed widget state before rendering so the first draw reflects stored data.
            if total_key not in st.session_state:
                st.session_state[total_key] = float(parsed_initial_total)

            new_total = st.number_input(
                "Total Amount",
                min_value=0.0,
                value=float(st.session_state[total_key]),
                step=0.01,
                key=total_key,
            )
            receipt_data["total_amount"] = f"{new_total:.2f}"

            # 3. VAT Amount
            current_vat = receipt_data.get("vat_amount", 0) or 0
            try:
                current_vat_float = float(current_vat)
            except (TypeError, ValueError):
                current_vat_float = 0.0
            vat_cols = st.columns([3, 2])
            with vat_cols[0]:
                vat_value = st.session_state.setdefault(
                    vat_key, float(current_vat_float)
                )
                new_vat = st.number_input(
                    "VAT Amount",
                    min_value=0.0,
                    value=vat_value,
                    step=0.01,
                    key=vat_key,
                )
                receipt_data["vat_amount"] = new_vat
            with vat_cols[1]:
                # Add breathing room so the button is vertically aligned with the input field.
                st.markdown("<div style='padding-top:11%'></div>", unsafe_allow_html=True)
                st.button(
                    "Calculate VAT",
                    key=receipt_widget_key(current_result, "calculate_vat"),
                    width="stretch",
                    help="Use the configured VAT rate to back-calculate VAT from the total amount.",
                    on_click=handle_calculate_vat,
                    kwargs={
                        "receipt_index": current_index,
                        "vat_state_key": vat_key,
                        "notice_state_key": vat_notice_key,
                    },
                )
            vat_notice = st.session_state.pop(vat_notice_key, None)
            if vat_notice:
                # Show the warning below the entire VAT control row for better visibility.
                st.warning(vat_notice)
            receipt_data["vat_amount"] = round(new_vat, 2)

            # 4. Transaction Date
            existing_date = receipt_data.get("transaction_date", "")
            parsed_date = parse_date_string(existing_date)
            date_display_value = parsed_date.isoformat() if parsed_date else (existing_date or "")
            new_date = st.text_input(
                "Transaction Date (YYYY-MM-DD)",
                value=date_display_value,
                key=date_key,
                placeholder="YYYY-MM-DD",
            )
            formatted_date = format_date_for_storage(new_date)
            if new_date.strip() and not formatted_date:
                st.warning("Unable to parse the transaction date. Please use YYYY-MM-DD format.")
            receipt_data["transaction_date"] = formatted_date if formatted_date else new_date.strip() or None

            # 5. Payment Mode
            payment_options = ["CASH", "CARD", "EFT", "Enter manually"]
            existing_payment = receipt_data.get("payment_mode") or ""
            standard_payments = {"CASH", "CARD", "EFT"}
            if existing_payment and isinstance(existing_payment, str) and existing_payment.upper() in standard_payments:
                default_index = payment_options.index(existing_payment.upper())
            else:
                default_index = payment_options.index("Enter manually")

            selected_payment_mode = st.selectbox(
                "Payment Mode",
                options=payment_options,
                index=default_index,
                key=payment_select_key,
            )

            manual_payment_mode = None
            if selected_payment_mode == "Enter manually":
                manual_default = (
                    existing_payment if existing_payment and existing_payment.upper() not in standard_payments else ""
                )
                manual_payment_mode = st.text_input(
                    "Payment Mode (manual entry)",
                    value=manual_default,
                    key=payment_manual_key,
                ).strip()
                receipt_data["payment_mode"] = manual_payment_mode
            else:
                receipt_data["payment_mode"] = selected_payment_mode

            # 6. Notes
            notes_value = st.text_area(
                "Notes",
                value=receipt_data.get("notes", ""),
                key=notes_key,
                height=80,
            )
            receipt_data["notes"] = notes_value.strip()

            # Items section
            st.subheader("Items")
            
            # Get item capture setting
            settings_manager = get_settings_manager()
            snapshot = settings_manager.get_settings_snapshot()
            enable_item_capture = snapshot.get("enable_item_capture", True)
            
            if enable_item_capture:
                # Ensure items list exists
                if "items" not in receipt_data:
                    receipt_data["items"] = []
                if not isinstance(receipt_data["items"], list):
                    receipt_data["items"] = []
                
                items = receipt_data["items"]
                
                # Display items in an editable table
                if items:
                    # Create DataFrame for display
                    items_data = []
                    for idx, item in enumerate(items):
                        items_data.append({
                            "Item Name": item.get("item_name", item.get("name", "")),
                            "Price": item.get("price", ""),
                            "COICOP": item.get("coicop", item.get("code", "")),
                            "COICOP Desc": item.get("coicop_desc", item.get("code_desc", "")),
                            "Confidence": item.get("confidence", item.get("prob", "")),
                        })
                    
                    items_df = pd.DataFrame(items_data)
                    st.dataframe(items_df, width="stretch", hide_index=False)
                    
                    # Delete item buttons
                    st.write("**Delete Items:**")
                    delete_cols = st.columns(min(len(items), 5))
                    for idx, item in enumerate(items):
                        col_idx = idx % 5
                        with delete_cols[col_idx]:
                            if st.button(
                                f"Delete #{idx+1}",
                                key=f"{receipt_widget_key(current_result, 'delete_item')}_{idx}",
                                width="stretch",
                            ):
                                items.pop(idx)
                                autosave_results()
                                st.rerun()
                else:
                    st.info("No items found. Add items below.")
                
                st.divider()
                
                # Add new item form
                with st.expander("➕ Add New Item", expanded=False):
                    new_item_cols = st.columns(2)
                    with new_item_cols[0]:
                        new_item_name = st.text_input(
                            "Item Name",
                            key=receipt_widget_key(current_result, "new_item_name"),
                        )
                        new_item_price = st.text_input(
                            "Price",
                            key=receipt_widget_key(current_result, "new_item_price"),
                            help="Enter price as number (e.g., 5.99)",
                        )
                        new_item_coicop = st.text_input(
                            "COICOP Code",
                            key=receipt_widget_key(current_result, "new_item_coicop"),
                        )
                    with new_item_cols[1]:
                        new_item_coicop_desc = st.text_input(
                            "COICOP Description",
                            key=receipt_widget_key(current_result, "new_item_coicop_desc"),
                        )
                        new_item_confidence = st.text_input(
                            "Confidence",
                            key=receipt_widget_key(current_result, "new_item_confidence"),
                            help="Optional: confidence score",
                        )
                    
                    if st.button(
                        "Add Item",
                        key=receipt_widget_key(current_result, "add_item"),
                        width="stretch",
                    ):
                        if new_item_name.strip():
                            new_item = {
                                "item_name": new_item_name.strip(),
                                "price": new_item_price.strip() if new_item_price.strip() else None,
                                "coicop": new_item_coicop.strip() if new_item_coicop.strip() else None,
                                "coicop_desc": new_item_coicop_desc.strip() if new_item_coicop_desc.strip() else None,
                                "confidence": new_item_confidence.strip() if new_item_confidence.strip() else None,
                            }
                            items.append(new_item)
                            receipt_data["items"] = items
                            autosave_results()
                            st.success(f"Added item: {new_item_name}")
                            st.rerun()
                        else:
                            st.warning("Please enter at least an item name.")
            else:
                st.info("Item capture has been disabled. Enable it in Application Settings to view and edit items.")
                    
    else:
        st.info("Upload receipt images to begin processing")
        
if __name__ == "__main__":
    main()
