"""use streamlit to create interface of receipt data entry."""

import base64
import inspect
import json
import os
import re
from datetime import date, datetime
from io import BytesIO


import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit.runtime.scriptrunner import RerunException, RerunData

from PIL import Image

# Force hosted-mode defaults when running the public Streamlit instance.
os.environ.setdefault("SCANNERAI_HOSTED_MODE", "1")

from scannerai._config.config import config
from scannerai.settings import SettingsManager
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
            use_container_width=True,
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

        save_settings = st.button(
            "Save Settings", use_container_width=True
        )

        if save_settings:
            validation_errors = []
            if hosted_mode and not export_passphrase.strip():
                validation_errors.append(
                    "Enter a passphrase so your settings can be encrypted for download."
                )
            if ocr_model in (1, 2) and not (openai_key_input.strip() or api_status.get("openai")):
                validation_errors.append(
                    "An OpenAI API key is required for GPT-based OCR models."
                )
            if ocr_model == 3:
                if not (gemini_key_input.strip() or api_status.get("gemini")):
                    validation_errors.append("A Gemini API key is required for Gemini OCR.")
                if not (
                    google_credentials_upload
                    or google_credentials_path.strip()
                    or existing_google_credentials_path.strip()
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
                use_container_width=True,
            )

        reset_clicked = st.button(
            "Reset settings to defaults",
            type="secondary",
            use_container_width=True,
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


def process_image(image_path, ocr_processor):
    """Process a single receipt image."""
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


def process_file_bytes(file_name, file_bytes, ocr_processor):
    """Process an in-memory file by writing it to a temporary location."""
    temp_path = f"temp_{file_name}"
    with open(temp_path, "wb") as temp_file:
        temp_file.write(file_bytes)
    try:
        return process_image(temp_path, ocr_processor)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def save_to_json(results, file_path):
    """Save results to JSON file."""
    serializable_results = [
        {
            "receipt_data": {
                **result["receipt_data"],
                "transaction_date": format_date_for_storage(
                    result["receipt_data"].get("transaction_date")
                ),
                "notes": result["receipt_data"].get("notes", ""),
            }
        }
        for result in results
    ]
    with open(file_path, "w") as json_file:
        json.dump(serializable_results, json_file, indent=4)

def save_to_csv(results, file_path):
    """Save results to CSV file."""
    rows = []
    for result in results:
        receipt_data = result["receipt_data"]
        if not receipt_data["items"]:
            rows.append(
                {
                    "item": "",
                    "code": "",
                    "code_desc": "",
                    "price": "",
                    "prob": "",
                    "shop_name": receipt_data["shop_name"],
                    "image_path": receipt_data.get("receipt_pathfile", ""),
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
                        "item": item.get("name", ""),
                        "code": item.get("code", ""),
                        "code_desc": item.get("code_desc", ""),
                        "price": item.get("price", ""),
                        "prob": item.get("prob", ""),
                        "shop_name": receipt_data["shop_name"],
                        "image_path": receipt_data.get("receipt_pathfile", ""),
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
        1 for entry in st.session_state.receipt_status if entry["status"] in {"processed", "skipped"}
    )
    st.session_state.process_counts = {"completed": completed, "total": total}


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
    }
    st.session_state["app_settings"] = resolved_settings

    openai_api_key = (
        settings_manager.get_api_key("openai") if settings_manager else config.openai_api_key
    )
    gemini_api_key = (
        settings_manager.get_api_key("gemini") if settings_manager else config.gemini_api_key
    )

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
    if "autosave_path" not in st.session_state:
        st.session_state.autosave_path = "receipt_autosave.json"
        
    # initialise OCR processor
    if "ocr_processor" not in st.session_state:
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

    # Sidebar for file upload and navigation
    with st.sidebar:
        st.header("Upload & Navigation")

        # File uploader - always visible
        # Use a key that can be reset to clear the uploader
        uploader_key = st.session_state.get("file_uploader_key", "file_uploader")
        uploaded_files = st.file_uploader(
            "Upload receipt images",
            type=["png", "jpg", "jpeg", "pdf"],
            accept_multiple_files=True,
            key=uploader_key,
        )

        # Process button - always visible when files are uploaded, but disabled during processing
        if uploaded_files:
            processing_active = st.session_state.get("processing_active", False)
            if st.button(
                "Process Uploaded Files",
                disabled=processing_active,
                use_container_width=True,
                help="Start processing uploaded files" if not processing_active else "Processing in progress..."
            ):
                files_data = [
                    {"name": file.name, "data": file.getvalue()}
                    for file in uploaded_files
                ]

                st.session_state.results = []
                st.session_state.current_index = 0
                st.session_state.receipt_status = []
                st.session_state.failed_receipts = []
                st.session_state.processing_queue = files_data
                st.session_state.process_counts = {
                    "completed": 0,
                    "total": len(files_data),
                }
                st.session_state.processing_active = True
                autosave_results()
                force_rerun()

        # Home button - show when results exist and processing is complete
        has_results = bool(st.session_state.results)
        processing_active = st.session_state.get("processing_active", False)
        processing_queue_empty = not st.session_state.get("processing_queue", [])
        
        # Show buttons when we have results and processing is not active (or queue is empty)
        processing_complete = not processing_active or (processing_queue_empty and has_results)
        if has_results and processing_complete:
            if st.button("🏠 Home", use_container_width=True, help="Return to home screen and clear current results"):
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
        
        # Navigation with state preservation - show when results exist and processing is complete
        if has_results and processing_complete:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Previous") and st.session_state.current_index > 0:
                    # Save current state before navigation
                    st.session_state.current_index -= 1
            with col2:
                if st.button("Next") and st.session_state.current_index < len(st.session_state.results) - 1:
                    # Save current state before navigation
                    st.session_state.current_index += 1
            st.write(f"Receipt {st.session_state.current_index + 1} of {len(st.session_state.results)}")

        # Export options - show when results exist and processing is complete
        if has_results and processing_complete:
            st.header("Export Data")
            export_format = st.selectbox("Export format", ["JSON", "CSV"])

            # Get item capture setting
            settings_manager = get_settings_manager()
            snapshot = settings_manager.get_settings_snapshot()
            enable_item_capture = snapshot.get("enable_item_capture", True)

            # Prepare export data
            if export_format == "JSON":
                import json
                serializable_results = []
                for result in st.session_state.results:
                    receipt_data = result["receipt_data"].copy()
                    
                    # Build export data with all fields
                    export_receipt_data = {
                        "shop_name": receipt_data.get("shop_name"),
                        "total_amount": receipt_data.get("total_amount"),
                        "vat_amount": receipt_data.get("vat_amount", 0),
                        "payment_mode": receipt_data.get("payment_mode"),
                        "transaction_date": format_date_for_storage(
                            receipt_data.get("transaction_date")
                        ),
                        "notes": receipt_data.get("notes", ""),
                        "receipt_pathfile": receipt_data.get("receipt_pathfile", ""),
                    }
                    
                    # Only include items if item capture is enabled
                    if enable_item_capture and "items" in receipt_data:
                        export_receipt_data["items"] = receipt_data["items"]
                    
                    serializable_results.append({
                        "receipt_data": export_receipt_data,
                        "file_name": result.get("file_name") or os.path.basename(receipt_data.get("receipt_pathfile", "")),
                    })
                
                export_data = json.dumps(serializable_results, indent=4, ensure_ascii=False)
                mime_type = "application/json"
                file_extension = "json"
            else:  # CSV
                rows = []
                for result in st.session_state.results:
                    receipt_data = result["receipt_data"]
                    file_name = result.get("file_name") or os.path.basename(receipt_data.get("receipt_pathfile", ""))
                    
                    # Base fields that are always included
                    base_row = {
                        "file_name": file_name,
                        "shop_name": receipt_data.get("shop_name", ""),
                        "total_amount": receipt_data.get("total_amount", ""),
                        "vat_amount": receipt_data.get("vat_amount", ""),
                        "payment_mode": receipt_data.get("payment_mode", ""),
                        "transaction_date": receipt_data.get("transaction_date", ""),
                        "notes": receipt_data.get("notes", ""),
                        "receipt_pathfile": receipt_data.get("receipt_pathfile", ""),
                    }
                    
                    # Include items only if item capture is enabled
                    if enable_item_capture and receipt_data.get("items"):
                        for item in receipt_data["items"]:
                            row = base_row.copy()
                            row.update({
                                "item": item.get("item_name", item.get("name", "")),
                                "code": item.get("coicop", item.get("code", "")),
                                "code_desc": item.get("coicop_desc", item.get("code_desc", "")),
                                "price": item.get("price", ""),
                                "prob": item.get("confidence", item.get("prob", "")),
                            })
                            rows.append(row)
                    elif enable_item_capture:
                        # Item capture enabled but no items - add empty item columns
                        base_row.update({
                            "item": "",
                            "code": "",
                            "code_desc": "",
                            "price": "",
                            "prob": "",
                        })
                        rows.append(base_row)
                    else:
                        # Items disabled - just include base fields (no item columns)
                        rows.append(base_row)
                
                df = pd.DataFrame(rows)
                export_data = df.to_csv(index=False)
                mime_type = "text/csv"
                file_extension = "csv"

            # Download button
            st.download_button(
                label=f"Download {export_format}",
                data=export_data,
                file_name=f"receipt_data.{file_extension}",
                mime=mime_type,
                use_container_width=True,
            )
        if st.session_state.processing_active and st.session_state.processing_queue:
            queue_entry = st.session_state.processing_queue[0]
            counts = st.session_state.process_counts
            total = max(counts["total"], 1)
            completed = counts["completed"]
            progress_ratio = completed / total
            progress_text = (
                f"Processing {queue_entry['name']} ({completed + 1}/{total})..."
            )
            progress_bar = st.progress(progress_ratio, text=progress_text)

            update_receipt_status(queue_entry["name"], "processing")

            try:
                result = process_file_bytes(
                    queue_entry["name"], queue_entry["data"], st.session_state.ocr_processor
                )
                if result:
                    result["receipt_data"]["processing_status"] = "processed"
                    st.session_state.results.append(result)
                    update_receipt_status(queue_entry["name"], "processed")
                else:
                    update_receipt_status(queue_entry["name"], "error", "No data returned.")
                    st.session_state.failed_receipts.append(
                        {
                            "name": queue_entry["name"],
                            "data": queue_entry["data"],
                            "error": "No data returned.",
                        }
                    )
            except Exception as exc:  # pragma: no cover
                update_receipt_status(queue_entry["name"], "error", str(exc))
                st.session_state.failed_receipts.append(
                    {
                        "name": queue_entry["name"],
                        "data": queue_entry["data"],
                        "error": str(exc),
                    }
                )
            finally:
                autosave_results()
                st.session_state.processing_queue.pop(0)

            counts = st.session_state.process_counts
            total = max(counts["total"], 1)
            updated_ratio = counts["completed"] / total
            status_text = f"Completed {counts['completed']} / {total}"
            progress_bar.progress(updated_ratio, text=status_text)

            if st.session_state.processing_queue:
                force_rerun()
            else:
                st.session_state.processing_active = False
                st.success("Processing complete.")
                # Force rerun to update sidebar buttons immediately
                st.rerun()

        # Only show "Completed */*" when processing is active or just completed (and on first image)
        counts = st.session_state.process_counts
        processing_active = st.session_state.get("processing_active", False)
        has_results_main = bool(st.session_state.results)
        is_first_image_main = st.session_state.current_index == 0
        
        if counts["total"] > 0 and (processing_active or (has_results_main and is_first_image_main)):
            st.markdown(
                f"<div style='text-align:center; font-weight:600;'>Completed {counts['completed']} / {counts['total']}</div>",
                unsafe_allow_html=True,
            )

        if st.session_state.receipt_status:
            st.write("Processing status:")
            for entry in st.session_state.receipt_status:
                status = entry["status"]
                file_name = entry["file"]
                message = entry.get("message")
                bullet = "🟢" if status == "processed" else ("⚠️" if status == "skipped" else "🔴")
                text = f"{bullet} {file_name} — {status.capitalize()}"
                if message and status not in {"processed", "skipped"}:
                    text += f" ({message})"
                st.write(text)

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
                        result = process_file_bytes(
                            failed["name"], failed["data"], st.session_state.ocr_processor
                        )
                        if result:
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

                viewer_html = f"""
                <style>
                    #{viewer_id} {{
                        width: 100%;
                        height: 600px;
                        overflow: hidden;
                        border: 1px solid #d9d9d9;
                        border-radius: 6px;
                        background: #f7f7f7;
                        position: relative;
                        cursor: grab;
                        touch-action: none;
                    }}
                    #{viewer_id} img {{
                        display: block;
                        transform-origin: 0 0;
                        user-select: none;
                        -webkit-user-drag: none;
                        transition: transform 0.08s ease-out;
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
                    <div id="{zoom_info_id}">50%</div>
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
                        let scale = 0.5;
                        let offsetX = 0;
                        let offsetY = 0;
                        let isPanning = false;
                        let startX = 0;
                        let startY = 0;
                        let startOffsetX = 0;
                        let startOffsetY = 0;
                        const activeTouches = new Map();
                        let pinchStartDistance = null;
                        let pinchStartScale = null;

                        const clamp = (value, min, max) => Math.min(Math.max(value, min), max);

                        const applyTransform = () => {{
                            img.style.transform = `translate(${{offsetX}}px, ${{offsetY}}px) scale(${{scale}})`;
                            if (info) {{
                                info.textContent = `${{Math.round(scale * 100)}}%`;
                            }}
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

                        const centerImage = () => {{
                            const rect = viewer.getBoundingClientRect();
                            const naturalWidth = img.naturalWidth || rect.width;
                            const naturalHeight = img.naturalHeight || rect.height;
                            const displayWidth = naturalWidth * scale;
                            const displayHeight = naturalHeight * scale;
                            offsetX = (rect.width - displayWidth) / 2;
                            const extraY = rect.height - displayHeight;
                            offsetY = extraY > 0 ? extraY / 2 : 0;
                            applyTransform();
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
                            scale = 0.5;
                            centerImage();
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
                            centerImage();
                        }};

                        if (img.complete) {{
                            initialize();
                        }} else {{
                            img.addEventListener("load", initialize, {{ once: true }});
                        }}

                        return () => {{
                            window.removeEventListener("mousemove", mouseMoveListener);
                            window.removeEventListener("mouseup", mouseUpListener);
                        }};
                    }})();
                </script>
                """

                components.html(viewer_html, height=620, width=None)

        with col2:
            st.subheader("Receipt Data")

            # Shop details
            receipt_data = current_result["receipt_data"]

            # Create unique keys for each input field
            shop_key = f"shop_name_{current_index}"
            total_key = f"total_amount_{current_index}"
            vat_key = f"vat_amount_{current_index}"
            date_key = f"transaction_date_{current_index}"
            payment_select_key = f"payment_mode_select_{current_index}"
            payment_manual_key = f"payment_mode_manual_{current_index}"
            notes_key = f"notes_{current_index}"

            # 1. Shop Name
            new_shop_name = st.text_input(
                "Shop Name",
                value=receipt_data["shop_name"],
                key=shop_key,
            )
            receipt_data["shop_name"] = new_shop_name

            # 2. Total Amount
            new_total = st.text_input(
                "Total Amount",
                value=format_currency_string(receipt_data.get("total_amount")),
                key=total_key,
            )
            parsed_total = parse_float(new_total)
            if parsed_total is not None:
                receipt_data["total_amount"] = f"{parsed_total:.2f}"
            else:
                receipt_data["total_amount"] = new_total.strip() or None

            # 3. VAT Amount
            current_vat = receipt_data.get("vat_amount", 0) or 0
            try:
                current_vat_float = float(current_vat)
            except (TypeError, ValueError):
                current_vat_float = 0.0
            new_vat = st.number_input(
                "VAT Amount",
                min_value=0.0,
                value=float(current_vat_float),
                step=0.01,
                key=vat_key,
            )
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
                    st.dataframe(items_df, use_container_width=True, hide_index=False)
                    
                    # Delete item buttons
                    st.write("**Delete Items:**")
                    delete_cols = st.columns(min(len(items), 5))
                    for idx, item in enumerate(items):
                        col_idx = idx % 5
                        with delete_cols[col_idx]:
                            if st.button(f"Delete #{idx+1}", key=f"delete_item_{current_index}_{idx}", use_container_width=True):
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
                        new_item_name = st.text_input("Item Name", key=f"new_item_name_{current_index}")
                        new_item_price = st.text_input("Price", key=f"new_item_price_{current_index}", help="Enter price as number (e.g., 5.99)")
                        new_item_coicop = st.text_input("COICOP Code", key=f"new_item_coicop_{current_index}")
                    with new_item_cols[1]:
                        new_item_coicop_desc = st.text_input("COICOP Description", key=f"new_item_coicop_desc_{current_index}")
                        new_item_confidence = st.text_input("Confidence", key=f"new_item_confidence_{current_index}", help="Optional: confidence score")
                    
                    if st.button("Add Item", key=f"add_item_{current_index}", use_container_width=True):
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
