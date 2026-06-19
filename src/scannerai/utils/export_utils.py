"""Helpers for receipt export filenames, rows, and ZIP payloads."""

from __future__ import annotations

import csv
import io
import json
import re
import unicodedata
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

UNKNOWN_STORE = "UnknownStore"
DATA_FILENAMES = {
    "CSV": "receipt_data.csv",
    "JSON": "receipt_data.json",
}

_DATE_FORMATS = (
    "%Y-%m-%d",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%Y/%m/%d",
    "%d %b %Y",
    "%d %B %Y",
    "%d.%m.%Y",
    "%m/%d/%Y",
    "%m-%d-%Y",
)


@dataclass(frozen=True)
class ReceiptExportFile:
    """A planned receipt file entry for the ZIP export."""

    result_index: int
    renamed_file_name: str
    zip_path: str
    original_file_name: str
    original_file_bytes: bytes | None


def sanitize_store_name(value: Any) -> str:
    """Return a filesystem-safe store token for exported receipt filenames."""
    if value is None:
        return UNKNOWN_STORE

    normalized = unicodedata.normalize("NFKD", str(value).strip())
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", ascii_value).strip("_")
    return cleaned or UNKNOWN_STORE


def _coerce_date(value: Any, fallback: date) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if value in (None, "", "null"):
        return fallback

    value_str = str(value).strip().replace("\\", "/")
    if not value_str:
        return fallback

    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(value_str, fmt).date()
        except ValueError:
            continue
    return fallback


def format_export_date(value: Any, fallback: date) -> str:
    """Return a receipt/export date token as ddmmyy."""
    return _coerce_date(value, fallback).strftime("%d%m%y")


def _original_file_name(result: Mapping[str, Any]) -> str:
    receipt_data = result.get("receipt_data", {}) or {}
    fallback_path = str(receipt_data.get("receipt_pathfile") or "")
    return str(
        result.get("original_file_name")
        or result.get("file_name")
        or Path(fallback_path).name
        or "receipt.bin"
    )


def _extension(file_name: str) -> str:
    suffix = Path(file_name).suffix
    if suffix and re.fullmatch(r"\.[A-Za-z0-9]+", suffix):
        return suffix
    return ".bin"


def _original_bytes(result: Mapping[str, Any]) -> bytes | None:
    payload = result.get("original_file_bytes")
    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, bytearray):
        return bytes(payload)
    return None


def build_receipt_file_plan(
    results: Sequence[Mapping[str, Any]],
    export_date: date | None = None,
) -> list[ReceiptExportFile]:
    """Build deterministic renamed receipt filenames for export results."""
    fallback_date = export_date or date.today()
    counters: dict[str, int] = {}
    planned_files: list[ReceiptExportFile] = []

    for index, result in enumerate(results):
        receipt_data = result.get("receipt_data", {}) or {}
        store_token = sanitize_store_name(receipt_data.get("shop_name"))
        date_token = format_export_date(
            receipt_data.get("transaction_date"), fallback_date
        )
        base_name = f"{store_token}{date_token}"
        counters[base_name] = counters.get(base_name, 0) + 1

        original_name = _original_file_name(result)
        renamed_name = f"{base_name}_{counters[base_name]}{_extension(original_name)}"
        planned_files.append(
            ReceiptExportFile(
                result_index=index,
                renamed_file_name=renamed_name,
                zip_path=f"receipts/{renamed_name}",
                original_file_name=original_name,
                original_file_bytes=_original_bytes(result),
            )
        )

    return planned_files


def _serializable(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value


def _base_receipt_data(receipt_data: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "shop_name": _serializable(receipt_data.get("shop_name")),
        "total_amount": _serializable(receipt_data.get("total_amount")),
        "vat_amount": _serializable(receipt_data.get("vat_amount", 0)),
        "payment_mode": _serializable(receipt_data.get("payment_mode")),
        "transaction_date": _serializable(receipt_data.get("transaction_date")),
        "notes": _serializable(receipt_data.get("notes", "")),
    }


def build_json_export(
    results: Sequence[Mapping[str, Any]],
    file_plan: Sequence[ReceiptExportFile],
    enable_item_capture: bool,
) -> str:
    """Build JSON export text using renamed receipt filenames."""
    payload = []
    for result, planned_file in zip(results, file_plan):
        receipt_data = result.get("receipt_data", {}) or {}
        export_receipt = _base_receipt_data(receipt_data)
        if enable_item_capture:
            export_receipt["items"] = receipt_data.get("items", []) or []

        payload.append(
            {
                "file_name": planned_file.renamed_file_name,
                "receipt_data": export_receipt,
            }
        )

    return json.dumps(payload, indent=4, ensure_ascii=False)


def _text(value: Any) -> Any:
    return "" if value is None else _serializable(value)


def _item_rows(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "item": item.get("item_name", item.get("name", "")),
        "code": item.get("coicop", item.get("code", "")),
        "code_desc": item.get("coicop_desc", item.get("code_desc", "")),
        "price": item.get("price", ""),
        "prob": item.get("confidence", item.get("prob", "")),
    }


def build_csv_export(
    results: Sequence[Mapping[str, Any]],
    file_plan: Sequence[ReceiptExportFile],
    enable_item_capture: bool,
) -> str:
    """Build CSV export text using renamed receipt filenames."""
    base_fields = [
        "file_name",
        "shop_name",
        "total_amount",
        "vat_amount",
        "payment_mode",
        "transaction_date",
        "notes",
    ]
    item_fields = ["item", "code", "code_desc", "price", "prob"]
    fieldnames = base_fields + (item_fields if enable_item_capture else [])
    rows: list[dict[str, Any]] = []

    for result, planned_file in zip(results, file_plan):
        receipt_data = result.get("receipt_data", {}) or {}
        base_row = {
            "file_name": planned_file.renamed_file_name,
            "shop_name": _text(receipt_data.get("shop_name", "")),
            "total_amount": _text(receipt_data.get("total_amount", "")),
            "vat_amount": _text(receipt_data.get("vat_amount", "")),
            "payment_mode": _text(receipt_data.get("payment_mode", "")),
            "transaction_date": _text(receipt_data.get("transaction_date", "")),
            "notes": _text(receipt_data.get("notes", "")),
        }

        items = receipt_data.get("items", []) or []
        if enable_item_capture and items:
            for item in items:
                row = base_row.copy()
                row.update(_item_rows(item if isinstance(item, Mapping) else {}))
                rows.append(row)
        elif enable_item_capture:
            row = base_row.copy()
            row.update({field: "" for field in item_fields})
            rows.append(row)
        else:
            rows.append(base_row)

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def build_export_data(
    results: Sequence[Mapping[str, Any]],
    export_format: str,
    enable_item_capture: bool,
    export_date: date | None = None,
) -> tuple[str, str, list[ReceiptExportFile]]:
    """Return export filename, text content, and receipt rename plan."""
    normalized_format = export_format.upper()
    if normalized_format not in DATA_FILENAMES:
        raise ValueError("export_format must be CSV or JSON.")

    file_plan = build_receipt_file_plan(results, export_date=export_date)
    if normalized_format == "JSON":
        content = build_json_export(results, file_plan, enable_item_capture)
    else:
        content = build_csv_export(results, file_plan, enable_item_capture)

    return DATA_FILENAMES[normalized_format], content, file_plan


def build_export_zip(
    results: Sequence[Mapping[str, Any]],
    export_format: str,
    enable_item_capture: bool,
    export_date: date | None = None,
) -> bytes:
    """Build a ZIP containing the selected export plus renamed receipt uploads."""
    data_filename, data_content, file_plan = build_export_data(
        results,
        export_format,
        enable_item_capture,
        export_date=export_date,
    )

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(data_filename, data_content.encode("utf-8"))
        for planned_file in file_plan:
            if planned_file.original_file_bytes is not None:
                zf.writestr(
                    planned_file.zip_path,
                    planned_file.original_file_bytes,
                )

    return buffer.getvalue()


def missing_original_file_names(
    file_plan: Iterable[ReceiptExportFile],
) -> list[str]:
    """Return original filenames that cannot be written to the ZIP."""
    return [
        planned_file.original_file_name
        for planned_file in file_plan
        if planned_file.original_file_bytes is None
    ]
