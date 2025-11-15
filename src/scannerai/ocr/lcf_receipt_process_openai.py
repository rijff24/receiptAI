"""use Tesseract + GPT-3 to extract structured information from an image."""

import ast
import json
import os
import re
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import pytesseract
from openai import OpenAI

from scannerai.utils.scanner_utils import (
    count_tokens_openai,
    merge_pdf_pages,
    read_api_key,
)


class LCFReceiptProcessOpenai:
    """class to extract text from image using OpenAI API."""

    def __init__(
        self,
        openai_api_key_path=None,
        tesseract_cmd_path=None,
        openai_api_key=None,
    ):
        """Initialize Openai API with credentials."""

        self.InitSuccess = False  # Initialize to False
        self.client = None  # Initialize to None
        if tesseract_cmd_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd_path

        api_key = openai_api_key or self._load_key_from_path(openai_api_key_path)
        if not api_key:
            print("WARNING: ChatGPT API key not supplied. OpenAI OCR disabled.")
            return

        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key)
        self.InitSuccess = True

    @staticmethod
    def _load_key_from_path(key_path):
        """Load an API key from a file path."""
        if not key_path:
            return None
        if not os.path.exists(key_path):
            print(f"WARNING: ChatGPT API key file does not exist: {key_path}")
            return None
        return read_api_key(key_path)

    def get_InitSuccess(self):
        """Return the initialization status."""
        return self.InitSuccess

    def estimate_tokens(self, messages):
        """Estimate token counts."""
        token_count = 0
        token_text = ""
        for message in messages:
            if isinstance(message["content"], list):
                for content in message["content"]:
                    token_text += count_tokens_openai(
                        "gpt-3.5-turbo", content["text"]
                    )
            else:
                token_count += count_tokens_openai(
                    "gpt-3.5-turbo", message["content"]
                )
        return token_count

    # Function to call OpenAI API and format the receipt information
    def extract_receipt_with_chatgpt(self, ocr_text, enable_price_count=False):
        """To call OpenAI API and format the receipt information."""
        prompt = (
            f"You are given OCR text from a retail receipt.\n"
            f'OCR TEXT:\n"""{ocr_text}"""\n'
            "Return a **valid JSON object** using double quotes, e.g. "
            '{"shop_name": "...", "items": [{"name": "...", "price": 0.00}], '
            '"total_amount": 0.00, "vat_amount": 0.00, "payment_mode": "...", '
            '"transaction_date": "YYYY-MM-DD"}.\n'
            "- Use numbers without thousands separators (e.g. 1219.53, not 1,219.53).\n"
            "- Ensure every item has a non-empty name; use an empty string if uncertain.\n"
            "- Use null for missing numeric values; if VAT is not shown use 0.0.\n"
            "- Classify payment_mode as 'CARD' whenever there is a card/POS slip (authorization codes, merchant copy, speed point, card machine references, etc.), even if other text mentions cash.\n"
            "- Only use 'CASH' when cash markers such as 'amount tendered' and 'change' appear; otherwise default uncertain cases to 'EFT'.\n"
            "- Only include the fields: shop_name, items (list of objects with name and price), total_amount, vat_amount, payment_mode, transaction_date.\n"
            "- Do not include any explanatory text—respond with JSON only."
        )

        # try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",  # Changed from GPT-4 to GPT-3.5-turbo
            messages=messages,
            temperature=0.5,
        )

        # Extract the response text
        # Print the response
        receipt_info = response.choices[0].message.content

        if enable_price_count:
            input_tokens = self.estimate_tokens(messages)
            output_tokens = count_tokens_openai("gpt-3.5-turbo", receipt_info)
            total_tokens = input_tokens + output_tokens
            print(f"Estimated input tokens: {input_tokens}")
            print(f"Output tokens: {output_tokens}")
            print(f"Estimated total tokens: {total_tokens}")

        return receipt_info

    @staticmethod
    def _normalise_amount(value):
        """Convert string amount with optional commas into float."""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        cleaned = value.strip().replace(",", "")
        try:
            return float(cleaned)
        except ValueError:
            return None

    def extract_vat_from_text(self, text):
        """Attempt to extract VAT amount directly from OCR text."""
        if not text:
            return None

        vat_patterns = [
            r"vat(?:\s*(?:amount|total|incl\.?|excl\.?)?)?\s*[:=]\s*([\d.,]+)",
            r"vat\s*([\d.,]+)",
            r"tax\s*[:=]\s*([\d.,]+)",
        ]

        for pattern in vat_patterns:
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            for match in matches:
                amount = self._normalise_amount(match)
                if amount is not None:
                    return round(amount, 2)

        return None

    @staticmethod
    def _normalise_payment_mode(value, raw_text=""):
        """Map payment descriptors into CARD/CASH/EFT categories."""
        candidates = []
        if value:
            candidates.append(str(value))
        if raw_text:
            candidates.append(str(raw_text))
        combined = " ".join(candidates).lower()

        if re.search(
            r"\b(card|credit|debit|visa|mastercard|amex|pos|speed\s*point|chip|tap|slip|authori[sz]ation|merchant copy|card machine)\b",
            combined,
        ):
            return "CARD"
        if re.search(
            r"\b(cash|change|tendered|notes|coins|amount\s+tendered|cash\s+tendered)\b",
            combined,
        ):
            return "CASH"
        if re.search(
            r"\b(branch|bank|transfer|eft|direct\s*deposit|cheque)\b",
            combined,
        ):
            return "EFT"
        return "EFT"

    @staticmethod
    def _normalise_date_string(value):
        """Convert various date formats to ISO string."""
        if not value:
            return None
        if isinstance(value, datetime):
            return value.date().isoformat()
        value_str = str(value).strip()
        if not value_str:
            return None

        normalized = value_str.replace("\\", "/").replace(".", "/")

        date_formats = [
            "%Y-%m-%d",
            "%d-%m-%Y",
            "%d/%m/%Y",
            "%Y/%m/%d",
            "%d %b %Y",
            "%d %B %Y",
            "%m/%d/%Y",
            "%m-%d-%Y",
        ]

        for fmt in date_formats:
            try:
                return datetime.strptime(normalized, fmt).date().isoformat()
            except ValueError:
                continue
        return None

    def extract_date_from_text(self, text):
        """Attempt to extract a transaction date from OCR text."""
        if not text:
            return None

        date_patterns = [
            r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",
            r"\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b",
            r"\b\d{1,2}\s+[A-Za-z]{3,}\s+\d{4}\b",
            r"\b[A-Za-z]{3,}\s+\d{1,2},\s+\d{4}\b",
        ]

        for pattern in date_patterns:
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            for match in matches:
                normalized = self._normalise_date_string(match)
                if normalized:
                    return normalized
        return None

    def parse_receipt_info(self, receipt_info):
        """Parse the input text to structured data."""
        receipt_data = {
            "shop_name": None,
            "items": [],
            "total": None,
            # "payment_mode": None,
            # "date": None,
            # "time": None,
        }

        shop_name = ""
        items_with_prices = []
        total_amount = ""
        payment_mode = ""

        lines = receipt_info.split("\n")
        for line in lines:
            line = line.replace("*", "")
            if re.search(
                "shop name", line, re.IGNORECASE
            ):  # line.startswith("1. Shop Name:"):
                shop_name = line.split(":")[1].strip()
            elif re.search(
                "list of items with their prices", line, re.IGNORECASE
            ):  # line.startswith("2. List of items with their prices:"):
                items_start = lines.index(line) + 1
                for item_line in lines[items_start:]:
                    if item_line.startswith("3."):
                        break
                    items_with_prices.append(item_line.strip())
            elif re.search(
                "Total amount paid", line, re.IGNORECASE
            ):  # line.startswith("3. Total amount paid:"):
                total_amount = line.split(":")[1].strip()
            elif re.search(
                "Payment mode", line, re.IGNORECASE
            ):  # line.startswith("4. Payment mode:"):
                payment_mode = line.split(":")[1].strip()

        receipt_data["shop_name"] = shop_name

        # extract item description and price
        item_regex = re.compile(r"\s*-\s*(.*?)\s*(?::\s*£(\d+\.\d{2}))?$")

        # List to hold dictionaries of each parsed item
        for item_price in items_with_prices:
            item_match = item_regex.match(item_price)
            item_description = None
            if item_match:
                item_description = item_match.group(1).strip()
                item_price = item_match.group(2)
                if item_price:
                    item_price = int(float(item_match.group(2).strip()) * 100)
                else:
                    item_price = None  # no price provided
            receipt_data["items"].append(
                {
                    "name": item_description,
                    "price": item_price,
                    "bounding_boxes": None,
                }
            )

        receipt_data["total"] = {"total": total_amount, "bounding_boxes": None}

        receipt_data["payment_mode"] = payment_mode

        return receipt_data

    def format_receipt_info(self, input_dict):
        """Convert dictionary style receipt data to the testing data style."""
        outputs_df = pd.DataFrame(
            columns=["image_relative_path", "shop_name", "recdesc", "amtpaid"]
        )

        # List to hold dictionaries of each parsed item
        rows = []
        for item in input_dict["items"]:
            # Create a dictionary for each row (item)
            row = {
                "image_relative_path": input_dict["receipt_pathfile"],
                "shop_name": input_dict["shop_name"],
                "recdesc": item["name"],
                "amtpaid": item["price"],
            }
            rows.append(row)

        # Use pd.concat to append all rows to the DataFrame at once
        if rows:
            outputs_df = pd.concat(
                [outputs_df, pd.DataFrame(rows)], ignore_index=True
            )

        return outputs_df

    def process_receipt(self, image_path, enable_price_count=False):
        """To extract structured data from input image."""
        # Load image
        file_extension = os.path.splitext(image_path)[1].lower()
        if file_extension in [".jpg", ".jpeg", ".png"]:
            image = cv2.imread(image_path)
        elif file_extension == ".pdf":
            image = merge_pdf_pages(image_path)

            if image:
                # convert to cv2 format
                image = np.array(image)
                image = image[:, :, ::-1].copy()
            else:
                raise ValueError("Failed to convert PDF to image")
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        if image is None:
            print(f"Error: Unable to read image at {image_path}")
            return None

        receipt_data = {
            "shop_name": None,
            "payment_mode": None,
            "items": [],  # or None, depending on how you want to handle it
            "receipt_pathfile": image_path,
        }

        if not self.get_InitSuccess():
            return receipt_data

        # pre-processing image
        if enable_price_count:
            # processed_image = preprocess_image(image)
            # TO ADD...
            processed_image = image
        else:
            processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ocr_text = pytesseract.image_to_string(processed_image)

        # Call the function
        receipt_info = self.extract_receipt_with_chatgpt(
            ocr_text, enable_price_count
        )
        print(receipt_info)
        # Parse the receipt_info
        # receipt_data = parse_receipt_info(receipt_info)
        if isinstance(receipt_info, str):
            cleaned_info = receipt_info.strip()
            # Remove thousands separators between digits to keep JSON valid
            cleaned_info = re.sub(r"(?<=\d),(?=\d{3}(\D|$))", "", cleaned_info)
            try:
                receipt_data = json.loads(cleaned_info)
            except json.JSONDecodeError:
                # Best-effort fallback: replace single quotes and try again
                normalized = re.sub(r"(?<=\d),(?=\d{3}(\D|$))", "", cleaned_info.replace("'", '"'))
                try:
                    receipt_data = json.loads(normalized)
                except json.JSONDecodeError:
                    # As a final fallback, attempt literal_eval
                    receipt_data = ast.literal_eval(normalized)
        else:
            receipt_data = receipt_info
        receipt_data.setdefault("vat_amount", 0)
        receipt_data.setdefault("transaction_date", None)
        receipt_data.setdefault("notes", "")

        if self._normalise_amount(receipt_data.get("vat_amount")) in (None, 0):
            vat_guess = self.extract_vat_from_text(ocr_text)
            if vat_guess is not None:
                receipt_data["vat_amount"] = vat_guess

        if not self._normalise_date_string(receipt_data.get("transaction_date")):
            date_guess = self.extract_date_from_text(ocr_text)
            if date_guess is not None:
                receipt_data["transaction_date"] = date_guess
        else:
            receipt_data["transaction_date"] = self._normalise_date_string(
                receipt_data["transaction_date"]
            )

        raw_context = receipt_info if isinstance(receipt_info, str) else json.dumps(receipt_info)
        receipt_data["payment_mode"] = self._normalise_payment_mode(
            receipt_data.get("payment_mode"),
            f"{raw_context}\n{ocr_text}",
        )

        receipt_data["receipt_pathfile"] = image_path

        return receipt_data


# Example usage
# image_pathfile = '/path/to/your/image.jpg'
# processor = LCFReceiptProcessOpenai()
# receipt_data = processor.process_receipt(image_pathfile)
# if receipt_data is not None:
#     print(receipt_data)
