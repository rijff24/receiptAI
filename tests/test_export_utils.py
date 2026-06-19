"""Tests for receipt export helpers."""

# ruff: noqa: D103

import csv
import io
import json
import zipfile
from datetime import date

from scannerai.utils.export_utils import (
    build_csv_export,
    build_export_zip,
    build_json_export,
    build_receipt_file_plan,
    sanitize_store_name,
)


def _result(
    *,
    shop_name="Pick n Pay",
    transaction_date="2026-06-19",
    original_file_name="receipt.jpg",
    original_file_bytes=b"image-bytes",
    items=None,
):
    return {
        "file_name": original_file_name,
        "original_file_name": original_file_name,
        "original_file_bytes": original_file_bytes,
        "receipt_data": {
            "shop_name": shop_name,
            "total_amount": "12.50",
            "vat_amount": 1.63,
            "payment_mode": "CARD",
            "transaction_date": transaction_date,
            "notes": "checked",
            "receipt_pathfile": "temp_should_not_export.jpg",
            "items": items or [],
        },
    }


def test_sanitize_store_name_for_safe_filename():
    assert sanitize_store_name(" Pick n Pay! ") == "Pick_n_Pay"
    assert sanitize_store_name("Muller Cafe") == "Muller_Cafe"
    assert sanitize_store_name("") == "UnknownStore"
    assert sanitize_store_name(None) == "UnknownStore"


def test_build_receipt_file_plan_increments_per_store_date_and_preserves_extension():
    results = [
        _result(original_file_name="first.JPG"),
        _result(original_file_name="second.pdf"),
        _result(shop_name="Other Store", original_file_name="third.png"),
    ]

    plan = build_receipt_file_plan(results, export_date=date(2026, 1, 1))

    assert [entry.renamed_file_name for entry in plan] == [
        "Pick_n_Pay190626_1.JPG",
        "Pick_n_Pay190626_2.pdf",
        "Other_Store190626_1.png",
    ]


def test_build_receipt_file_plan_uses_missing_data_fallbacks_and_bin_extension():
    result = _result(
        shop_name="",
        transaction_date=None,
        original_file_name="receipt",
        original_file_bytes=b"raw",
    )

    plan = build_receipt_file_plan([result], export_date=date(2026, 1, 2))

    assert plan[0].renamed_file_name == "UnknownStore020126_1.bin"


def test_json_export_uses_renamed_file_name_and_removes_file_locations():
    result = _result(items=[{"item_name": "Milk", "price": "20.00"}])
    plan = build_receipt_file_plan([result], export_date=date(2026, 1, 1))

    export_json = build_json_export([result], plan, enable_item_capture=True)
    payload = json.loads(export_json)

    assert payload[0]["file_name"] == "Pick_n_Pay190626_1.jpg"
    assert "receipt_pathfile" not in payload[0]["receipt_data"]
    assert payload[0]["receipt_data"]["items"][0]["item_name"] == "Milk"


def test_csv_export_removes_file_locations_and_respects_item_capture_toggle():
    result = _result(items=[{"item_name": "Milk", "price": "20.00"}])
    plan = build_receipt_file_plan([result], export_date=date(2026, 1, 1))

    export_csv = build_csv_export([result], plan, enable_item_capture=False)
    rows = list(csv.DictReader(io.StringIO(export_csv)))

    assert rows[0]["file_name"] == "Pick_n_Pay190626_1.jpg"
    assert "receipt_pathfile" not in rows[0]
    assert "image_path" not in rows[0]
    assert "item" not in rows[0]


def test_csv_export_includes_item_columns_when_item_capture_enabled():
    result = _result(items=[{"item_name": "Milk", "price": "20.00"}])
    plan = build_receipt_file_plan([result], export_date=date(2026, 1, 1))

    export_csv = build_csv_export([result], plan, enable_item_capture=True)
    rows = list(csv.DictReader(io.StringIO(export_csv)))

    assert rows[0]["item"] == "Milk"
    assert rows[0]["price"] == "20.00"


def test_zip_export_contains_selected_export_and_original_bytes():
    results = [
        _result(original_file_name="receipt.jpg", original_file_bytes=b"jpg-bytes"),
        _result(original_file_name="receipt.pdf", original_file_bytes=b"pdf-bytes"),
    ]

    zip_bytes = build_export_zip(
        results,
        "CSV",
        enable_item_capture=True,
        export_date=date(2026, 1, 1),
    )

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = set(zf.namelist())
        assert "receipt_data.csv" in names
        assert "receipts/Pick_n_Pay190626_1.jpg" in names
        assert "receipts/Pick_n_Pay190626_2.pdf" in names
        assert zf.read("receipts/Pick_n_Pay190626_1.jpg") == b"jpg-bytes"
        assert zf.read("receipts/Pick_n_Pay190626_2.pdf") == b"pdf-bytes"
