"""Use GPT4 vision API to extract structured information from an image."""

import base64
import io
import json
import os
import re
from datetime import datetime

from openai import OpenAI
from PIL import Image

from scannerai.utils.scanner_utils import (
    count_tokens_openai,
    estimate_image_tokens_openai,
    merge_pdf_pages,
    read_api_key,
)


class LCFReceiptProcessGPT4Vision:
    """Class to extract text from image using OpenAI ChatGPT4 vision API."""

    def __init__(self, openai_api_key_path=None, openai_api_key=None):
        """Initialize Openai API with credentials."""

        self.InitSuccess = False  # Initialize to False

        self.client = None

        api_key = openai_api_key or self._load_key_from_path(openai_api_key_path)
        if not api_key:
            print("WARNING: ChatGPT API key not supplied. GPT-4 Vision disabled.")
            return

        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key)

        # prompt is same for calling, so initialised
        self.prompt = (
            "Extract data from this receipt image and respond with a JSON object using double quotes only.\n"
            "Required structure:\n"
            "{\n"
            '  \"shop_name\": \"...\",\n'
            '  \"items\": [{\"name\": \"...\", \"price\": 0.00}],\n'
            '  \"total_amount\": 0.00,\n'
            '  \"vat_amount\": 0.00,\n'
            '  \"payment_mode\": \"...\",\n'
            '  \"transaction_date\": \"YYYY-MM-DD\"\n'
            "}\n"
            "- Use 0.0 when VAT is missing.\n"
            "- Use null for unknown numeric values.\n"
            "- Prefer ISO dates (YYYY-MM-DD); use null if unsure.\n"
            "- If the receipt clearly includes a card machine slip (card machine wording, POS references, merchant copies, authorization codes, etc.) classify payment_mode as \"CARD\" even if the text elsewhere mentions cash.\n"
            "- Only classify payment_mode as \"CASH\" when the receipt shows cash specific markers such as amount tendered + change due; otherwise default uncertain cases to \"EFT\".\n"
            "- Do not include explanations—return JSON only."
        )
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

    @staticmethod
    def encode_image(image):
        """Encode an image to the required format of input for gpt4 vision API."""

        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def estimate_vision_tokens(self, messages, image_width, image_height):
        """Estimate vision api tokens."""
        token_count = 0
        token_text = 0
        token_image = 0
        for message in messages:
            if isinstance(message["content"], list):
                for content in message["content"]:
                    if content["type"] == "text":
                        token_text += count_tokens_openai(
                            "gpt-4o-mini", content["text"]
                        )
                        token_count += count_tokens_openai(
                            "gpt-4o-mini", content["text"]
                        )
                    elif content["type"] == "image_url":
                        token_image += estimate_image_tokens_openai(
                            image_width, image_height
                        )
                        token_count += estimate_image_tokens_openai(
                            image_width, image_height
                        )
            else:
                token_count += count_tokens_openai(
                    "gpt-4o-mini", message["content"]
                )
        return token_count, token_text, token_image

    @staticmethod
    def _normalise_amount(value):
        """Convert value that may include commas/text into a rounded float."""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return round(float(value), 2)
        cleaned = str(value).strip().replace(",", "")
        match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
        if match:
            try:
                return round(float(match.group()), 2)
            except ValueError:
                return None
        return None

    @staticmethod
    def _normalise_date_string(value):
        """Convert various date formats to ISO string (YYYY-MM-DD)."""
        if not value:
            return None
        if isinstance(value, datetime):
            return value.date().isoformat()
        value_str = str(value).strip()
        if not value_str or value_str.upper() == "YYYY-MM-DD":
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

    @staticmethod
    def extract_vat_from_text(text):
        """Attempt to extract VAT amount directly from OCR response text."""
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
                amount = LCFReceiptProcessGPT4Vision._normalise_amount(match)
                if amount is not None:
                    return amount
        return None

    @staticmethod
    def extract_date_from_text(text):
        """Attempt to extract a transaction date from OCR response text."""
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
                normalized = LCFReceiptProcessGPT4Vision._normalise_date_string(match)
                if normalized:
                    return normalized
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
        if re.search(r"\b(branch|bank|transfer|eft|direct\s*deposit|cheque)\b", combined):
            return "EFT"
        return "EFT"

    def process_receipt(self, image_path, enable_price_count=False):
        """Process receipt."""
        file_extension = os.path.splitext(image_path)[1].lower()
        if file_extension in [".jpg", ".jpeg", ".png"]:
            with Image.open(image_path) as image:
                base64_image = LCFReceiptProcessGPT4Vision.encode_image(image)
        elif file_extension == ".pdf":
            image = merge_pdf_pages(image_path)

            if image:
                base64_image = LCFReceiptProcessGPT4Vision.encode_image(image)
            else:
                raise ValueError("Failed to convert PDF to image")
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        receipt_data = {
            "shop_name": None,
            "payment_mode": None,
            "items": [],  # or None, depending on how you want to handle it
            "receipt_pathfile": image_path,
        }

        if self.get_InitSuccess():
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ]

            max_tokens = 1000
            response = self.client.chat.completions.create(
                model="gpt-4o-mini", messages=messages, max_tokens=max_tokens
            )
            receipt_info = response.choices[0].message.content

            if response.choices[0].finish_reason == "length":
                print(
                    "ERROR: THE RESULTS EXCEEDED the max_token, so be partically cut off!"
                )
            else:
                structured_response = receipt_info
                if isinstance(structured_response, str):
                    lpos = structured_response.find("{")
                    if lpos != -1:
                        structured_response = structured_response[lpos:]
                    rpos = structured_response.rfind("}")
                    if rpos != -1:
                        structured_response = structured_response[: rpos + 1]
                    try:
                        receipt_data = json.loads(structured_response)
                    except json.JSONDecodeError:
                        receipt_data = {}
                else:
                    receipt_data = structured_response

                if not isinstance(receipt_data, dict):
                    receipt_data = {}

                receipt_data.setdefault("shop_name", None)
                receipt_data.setdefault("items", [])
                receipt_data.setdefault("total_amount", None)
                receipt_data.setdefault("vat_amount", 0)
                receipt_data.setdefault("payment_mode", None)
                receipt_data.setdefault("transaction_date", None)

                normalised_vat = self._normalise_amount(receipt_data.get("vat_amount"))
                if normalised_vat is None or normalised_vat == 0:
                    vat_guess = self.extract_vat_from_text(receipt_info)
                    normalised_vat = vat_guess if vat_guess is not None else 0.0
                receipt_data["vat_amount"] = normalised_vat

                normalised_date = self._normalise_date_string(
                    receipt_data.get("transaction_date")
                )
                if not normalised_date:
                    date_guess = self.extract_date_from_text(receipt_info)
                    normalised_date = date_guess
                receipt_data["transaction_date"] = normalised_date

                receipt_data["payment_mode"] = self._normalise_payment_mode(
                    receipt_data.get("payment_mode"), receipt_info
                )

                # Ensure item prices are numbers when possible
                items = receipt_data.get("items") or []
                for item in items:
                    if isinstance(item, dict) and "price" in item:
                        normalised_price = self._normalise_amount(item.get("price"))
                        item["price"] = normalised_price

        receipt_data["receipt_pathfile"] = image_path

        if enable_price_count:
            width, height = image.size
            input_tokens, input_tokens_text, input_tokens_image = (
                self.estimate_vision_tokens(messages, width, height)
            )

            output_tokens = count_tokens_openai("gpt-4o-mini", receipt_info)
            total_tokens = input_tokens + output_tokens
            print(
                f"Estimated input tokens: {input_tokens}, text tokens: {input_tokens_text}, image tokens: {input_tokens_image}"
            )
            print(f"Output tokens: {output_tokens}")
            print(f"Estimated total tokens: {total_tokens}")

        return receipt_data


# Example usage
# image_pathfile = '/path/to/your/image.jpg'
# processor = LCFReceiptProcessGPT4Vision()
# receipt_data = processor.process_receipt(image_pathfile)
# if receipt_data is not None:
#     print(json.dumps(receipt_data, indent=2))
