"""Class to hold config parameters."""

from __future__ import annotations

import os
from typing import Any, Dict

from dotenv import dotenv_values

try:
    from scannerai.settings import SettingsManager
except ImportError:  # pragma: no cover - optional dependency during install
    SettingsManager = None  # type: ignore

SETTINGS_KEY_MAP = {
    "debug_mode": "DEBUG_MODE",
    "enable_preprocessing": "ENABLE_PREPROCESSING",
    "save_processed_image": "SAVE_PROCESSED_IMAGE",
    "enable_price_count": "ENABLE_PRICE_COUNT",
    "ocr_model": "OCR_MODEL",
    "classifier_model_path": "CLASSIFIER_MODEL_PATH",
    "label_encoder_path": "LABEL_ENCODER_PATH",
    "tesseract_cmd_path": "TESSERACT_CMD_PATH",
    "google_credentials_path": "GOOGLE_CREDENTIALS_PATH",
}


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _get_value(key: str, file_values: Dict[str, Any], default: Any = None) -> Any:
    """Return a value prioritising environment variables over file values."""
    if key in os.environ:
        return os.environ.get(key)
    return file_values.get(key, default)


def _load_settings_manager_overrides() -> Dict[str, Any]:
    """Load overrides from the in-app settings manager, if available."""
    if not SettingsManager:
        return {}
    try:
        manager = SettingsManager()
    except Exception:
        return {}

    overrides = manager.export_settings()
    overrides["OPENAI_API_KEY"] = manager.get_api_key("openai")
    overrides["GEMINI_API_KEY"] = manager.get_api_key("gemini")
    overrides["GOOGLE_API_KEY"] = manager.get_api_key("google")
    return overrides


def load_config(config_file: str):
    """Load configuration from the legacy text file and settings manager."""
    file_values = dotenv_values(config_file) if os.path.exists(config_file) else {}

    config = {
        "DEBUG_MODE": _coerce_bool(_get_value("DEBUG_MODE", file_values, False)),
        "ENABLE_PREPROCESSING": _coerce_bool(
            _get_value("ENABLE_PREPROCESSING", file_values, False)
        ),
        "SAVE_PROCESSED_IMAGE": _coerce_bool(
            _get_value("SAVE_PROCESSED_IMAGE", file_values, False)
        ),
        "ENABLE_PRICE_COUNT": _coerce_bool(
            _get_value("ENABLE_PRICE_COUNT", file_values, False)
        ),
        "OCR_MODEL": _coerce_int(_get_value("OCR_MODEL", file_values, 2), 2),  # Default to GPT-4 Vision
        "CLASSIFIER_MODEL_PATH": _get_value("CLASSIFIER_MODEL_PATH", file_values),
        "LABEL_ENCODER_PATH": _get_value("LABEL_ENCODER_PATH", file_values),
        "GEMINI_API_KEY_PATH": _get_value("GEMINI_API_KEY_PATH", file_values),
        "OPENAI_API_KEY_PATH": _get_value("OPENAI_API_KEY_PATH", file_values),
        "GOOGLE_CREDENTIALS_PATH": _get_value("GOOGLE_CREDENTIALS_PATH", file_values),
        "TESSERACT_CMD_PATH": _get_value("TESSERACT_CMD_PATH", file_values),
        "OPENAI_API_KEY": None,
        "GEMINI_API_KEY": None,
        "GOOGLE_API_KEY": None,
    }

    overrides = _load_settings_manager_overrides()
    for settings_key, config_key in SETTINGS_KEY_MAP.items():
        if settings_key in overrides and overrides[settings_key] is not None:
            value = overrides[settings_key]
            if config_key == "OCR_MODEL":
                config[config_key] = _coerce_int(value, config[config_key])
            elif isinstance(config[config_key], bool):
                config[config_key] = _coerce_bool(value, config[config_key])
            else:
                config[config_key] = value

    for direct_key in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        if overrides.get(direct_key):
            config[direct_key] = overrides[direct_key]

    return config


# Create a Config class to handle configuration
class Config:
    """Class to handle configuration."""

    _instance = None
    _config = None

    def __new__(cls, config_file):
        """Create a configuration instance with config_file."""
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._config = load_config(config_file)
        return cls._instance

    @property
    def debug_mode(self):
        """Get DEBUG_MODE."""
        return self._config["DEBUG_MODE"]

    @property
    def enable_preprocessing(self):
        """Get ENABLE_PREPROCESSING."""
        return self._config["ENABLE_PREPROCESSING"]

    @property
    def save_processed_image(self):
        """Get SAVE_PROCESSED_IMAGE."""
        return self._config["SAVE_PROCESSED_IMAGE"]

    @property
    def enable_price_count(self):
        """Get ENABLE_PRICE_COUNT."""
        return self._config["ENABLE_PRICE_COUNT"]

    @property
    def ocr_model(self):
        """Get OCR_MODEL."""
        return self._config["OCR_MODEL"]

    @property
    def classifier_model_path(self):
        """Get CLASSIFIER_MODEL_PATH."""
        return self._config["CLASSIFIER_MODEL_PATH"]

    @property
    def label_encoder_path(self):
        """Get LABEL_ENCODER_PATH."""
        return self._config["LABEL_ENCODER_PATH"]

    @property
    def gemini_api_key_path(self):
        """Get GEMINI_API_KEY_PATH."""

        return self._config["GEMINI_API_KEY_PATH"]

    @property
    def openai_api_key_path(self):
        """Get OPENAI_API_KEY_PATH."""
        return self._config["OPENAI_API_KEY_PATH"]

    @property
    def open_api_key_path(self):
        """Backward compatible alias for OPENAI_API_KEY_PATH."""
        return self._config["OPENAI_API_KEY_PATH"]

    @property
    def tesseract_cmd_path(self):
        """Get TESSERACT_CMD_PATH."""
        return self._config["TESSERACT_CMD_PATH"]

    @property
    def google_credentials_path(self):
        """Get GOOGLE_CREDENTIALS_PATH."""
        return self._config["GOOGLE_CREDENTIALS_PATH"]

    @property
    def openai_api_key(self):
        """Get decrypted OpenAI API key when stored via settings."""
        return self._config.get("OPENAI_API_KEY")

    @property
    def gemini_api_key(self):
        """Get decrypted Gemini API key when stored via settings."""
        return self._config.get("GEMINI_API_KEY")

    @property
    def google_api_key(self):
        """Get decrypted Google API key when stored via settings."""
        return self._config.get("GOOGLE_API_KEY")


# Create a global instance
CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), "config.txt")
config = Config(CONFIG_FILE_PATH)

# Usage example:
# from scannerai.config.config import config
# if config.debug_mode:
#     print("Debug mode is enabled")
