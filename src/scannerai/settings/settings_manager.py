"""Utilities for managing local ScannerAI settings securely."""

from __future__ import annotations

import json
import os
import platform
import stat
from pathlib import Path
from typing import Any, Dict, Optional

from cryptography.fernet import Fernet, InvalidToken

try:  # pragma: no cover - optional dependency
    import keyring
except ImportError:  # pragma: no cover - fall back to file-based key
    keyring = None  # type: ignore


def _default_settings_dir() -> Path:
    """Return the appropriate cross-platform settings directory."""
    system = platform.system().lower()
    home = Path.home()

    if system == "windows":
        base = Path(os.getenv("APPDATA", home / "AppData" / "Roaming"))
    elif system == "darwin":
        base = home / "Library" / "Application Support"
    else:
        base = Path(os.getenv("XDG_CONFIG_HOME", home / ".config"))

    settings_dir = base / "ScannerAI"
    settings_dir.mkdir(parents=True, exist_ok=True)
    return settings_dir


def _get_default_training_files() -> tuple[str, str]:
    """Get default paths to training files in the package."""
    # Get the path to this file, then navigate to the trainedModels directory
    current_file = Path(__file__)
    # settings_manager.py is in src/scannerai/settings/
    # trainedModels is in src/scannerai/classifiers/trainedModels/
    package_root = current_file.parent.parent.parent
    trained_models_dir = package_root / "classifiers" / "trainedModels"
    
    classifier_path = str((trained_models_dir / "LRCountVectorizer.sav").resolve())
    encoder_path = str((trained_models_dir / "encoder.pkl").resolve())
    
    return classifier_path, encoder_path


class SettingsManager:
    """Persist user settings and API keys locally with optional encryption."""

    SETTINGS_FILENAME = "user_settings.json"
    KEY_FILENAME = ".scannerai.key"
    KEYRING_SERVICE = "ScannerAI"
    KEYRING_USER = "settings"

    @classmethod
    def get_defaults(cls) -> Dict[str, Any]:
        """Get default settings with computed default training file paths."""
        default_classifier, default_encoder = _get_default_training_files()
        return {
            "ocr_model": 2,  # Default to GPT-4 Vision
            "debug_mode": False,
            "enable_preprocessing": False,
            "save_processed_image": False,
            "enable_price_count": False,
            "classifier_model_path": default_classifier,
            "label_encoder_path": default_encoder,
            "tesseract_cmd_path": "",
            "google_credentials_path": "",
            "api_keys": {
                "openai": None,
                "gemini": None,
                "google": None,
            },
        }

    def __init__(self, settings_dir: Optional[Path] = None) -> None:
        self.settings_dir = settings_dir or _default_settings_dir()
        self.settings_path = self.settings_dir / self.SETTINGS_FILENAME
        self.key_path = self.settings_dir / self.KEY_FILENAME
        self._fernet = self._build_fernet()
        self._settings: Dict[str, Any] = {}
        self._load_settings()

    # --------------------------------------------------------------------- #
    # Persistence helpers
    # --------------------------------------------------------------------- #
    def _load_settings(self) -> None:
        """Load settings from disk, merging with defaults."""
        defaults = self.get_defaults()
        data: Dict[str, Any] = json.loads(json.dumps(defaults))
        file_data: Dict[str, Any] = {}

        if self.settings_path.exists():
            try:
                with self.settings_path.open("r", encoding="utf-8") as handle:
                    file_data = json.load(handle)
            except (json.JSONDecodeError, OSError):
                file_data = {}

        for key, default_value in defaults.items():
            if key == "api_keys":
                data["api_keys"].update(file_data.get("api_keys", {}))
            else:
                if key in file_data:
                    data[key] = file_data[key]
                else:
                    data[key] = default_value

        for key, value in file_data.items():
            if key not in data:
                data[key] = value

        self._settings = data

    def save_settings(self) -> None:
        """Persist current settings to disk."""
        serialisable = json.loads(json.dumps(self._settings))
        with self.settings_path.open("w", encoding="utf-8") as handle:
            json.dump(serialisable, handle, indent=2)
        self._harden_permissions(self.settings_path)

    def get_settings_snapshot(self) -> Dict[str, Any]:
        """Return a copy of settings with secrets redacted."""
        snapshot = json.loads(json.dumps(self._settings))
        for key, value in snapshot.get("api_keys", {}).items():
            snapshot["api_keys"][key] = bool(value)
        return snapshot

    def export_settings(self) -> Dict[str, Any]:
        """Return a deep copy of raw settings (includes encrypted api_keys)."""
        return json.loads(json.dumps(self._settings))

    def update_values(self, updates: Dict[str, Any]) -> None:
        """Update multiple scalar settings at once."""
        for key, value in updates.items():
            if key == "api_keys":
                continue
            self._settings[key] = value
        self.save_settings()

    def secure_file(self, path: Path) -> None:
        """Apply restrictive permissions to a file created outside the manager."""
        self._harden_permissions(path)

    # --------------------------------------------------------------------- #
    # Generic accessors
    # --------------------------------------------------------------------- #
    def get_value(self, key: str, default: Any = None) -> Any:
        """Retrieve a scalar setting."""
        defaults = self.get_defaults()
        return self._settings.get(key, default if default is not None else defaults.get(key))

    def set_value(self, key: str, value: Any) -> None:
        """Set a scalar setting and save."""
        if key == "api_keys":
            raise ValueError("Use set_api_key for API key updates.")
        self.update_values({key: value})

    # --------------------------------------------------------------------- #
    # API key encryption helpers
    # --------------------------------------------------------------------- #
    def set_api_key(self, provider: str, api_key: Optional[str]) -> None:
        """Encrypt and store an API key for a provider."""
        encrypted = self._encrypt(api_key) if api_key else None
        self._settings.setdefault("api_keys", {})[provider] = encrypted
        self.save_settings()

    def get_api_key(self, provider: str) -> Optional[str]:
        """Retrieve and decrypt an API key."""
        encrypted = self._settings.get("api_keys", {}).get(provider)
        if not encrypted:
            return None
        return self._decrypt(encrypted)

    def has_api_key(self, provider: str) -> bool:
        """Return True if an API key is stored for the provider."""
        return bool(self._settings.get("api_keys", {}).get(provider))

    # --------------------------------------------------------------------- #
    # Encryption primitives
    # --------------------------------------------------------------------- #
    def _build_fernet(self) -> Fernet:
        """Initialise a Fernet instance using keyring or a local key file."""
        key: Optional[str] = None
        if keyring:
            try:  # pragma: no cover - depends on system keyring
                key = keyring.get_password(self.KEYRING_SERVICE, self.KEYRING_USER)
            except keyring.errors.KeyringError:
                key = None

        if not key and self.key_path.exists():
            key = self.key_path.read_text(encoding="utf-8").strip()

        if not key:
            key = Fernet.generate_key().decode("utf-8")
            if keyring:
                try:  # pragma: no cover - depends on system keyring
                    keyring.set_password(self.KEYRING_SERVICE, self.KEYRING_USER, key)
                except keyring.errors.KeyringError:
                    self._persist_key_file(key)
            else:
                self._persist_key_file(key)
        return Fernet(key.encode("utf-8"))

    def _persist_key_file(self, key: str) -> None:
        """Persist encryption key to file as a fallback."""
        self.key_path.write_text(key, encoding="utf-8")
        self._harden_permissions(self.key_path)

    def _encrypt(self, value: Optional[str]) -> Optional[str]:
        """Encrypt a value using Fernet."""
        if value in (None, ""):
            return None
        return self._fernet.encrypt(value.encode("utf-8")).decode("utf-8")

    def _decrypt(self, token: str) -> Optional[str]:
        """Decrypt a value using Fernet."""
        if not token:
            return None
        try:
            return self._fernet.decrypt(token.encode("utf-8")).decode("utf-8")
        except InvalidToken:
            return None

    @staticmethod
    def _harden_permissions(path: Path) -> None:
        """Ensure file permissions are user-only where supported."""
        try:
            path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        except PermissionError:  # pragma: no cover - windows may not support chmod
            pass


