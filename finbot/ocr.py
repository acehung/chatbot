from __future__ import annotations

import base64
import json
import os
import re
from datetime import date
from pypdf import PdfReader
from openai import OpenAI


def extract_invoice_text(uploaded) -> str:
    if uploaded and uploaded.name.lower().endswith(".pdf"):
        try:
            reader = PdfReader(uploaded)
            page_text = [page.extract_text() or "" for page in reader.pages]
            if all(text.strip() for text in page_text):
                return "\n".join(page_text).strip()
            # Scanned pages have no text layer. Render only those pages and OCR
            # them when PyMuPDF, Pillow, and Tesseract are available.
            try:
                import fitz
                import pytesseract
                from PIL import Image
                from io import BytesIO

                document = fitz.open(stream=uploaded.getvalue(), filetype="pdf")
                pages = []
                for index, page in enumerate(document):
                    if page_text[index].strip():
                        pages.append(page_text[index])
                        continue
                    pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                    image = Image.open(BytesIO(pixmap.tobytes("png")))
                    pages.append(pytesseract.image_to_string(image, lang="vie+eng"))
                return "\n".join(pages).strip()
            except (ImportError, OSError, RuntimeError):
                return ""
        except Exception:
            return ""
    return ""


def _normalize_date(raw: str) -> str:
    m = re.match(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{4})$", raw.strip())
    if m:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return date(year, month, day).isoformat()
        except ValueError:
            pass
    m_iso = re.match(r"^(\d{4})[/-](\d{1,2})[/-](\d{1,2})$", raw.strip())
    if m_iso:
        year, month, day = int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3))
        try:
            return date(year, month, day).isoformat()
        except ValueError:
            pass
    return date.today().isoformat()


def parse_invoice(text: str) -> dict:
    patterns = {
        "invoice_id": r"(?:số hóa đơn|số\s*hđ|ký hiệu\s*số|số|invoice(?:\s*(?:no|number|id))?)\s*[:#-]?\s*([A-Z0-9/-]{3,})",
        "tax_id": r"(?:mã số thuế|mst|tax\s*id)\s*[:#-]?\s*([0-9-]{8,14})",
        "issue_date": r"(?:ngày lập|ngày\s*hóa\s*đơn|ngày|date)\s*[:#-]?\s*(\d{1,4}[/-]\d{1,2}[/-]\d{1,4})",
        "customer_name": r"(?:đơn vị bán|đơn vị mua|khách hàng|tên đơn vị|công ty|customer|vendor)\s*[:#-]?\s*([^\n\r]{3,60})",
    }
    result: dict = {}
    for field, pattern in patterns.items():
        m = re.search(pattern, text, re.I)
        if m:
            val = m.group(1).strip()
            if field == "issue_date":
                val = _normalize_date(val)
            result[field] = val

    amounts = re.findall(r"(?:tổng tiền thanh toán|tổng thanh toán|total|tổng cộng)\s*[:：]?\s*([\d.,]+)", text, re.I)
    vat = re.findall(r"(?:tiền thuế\s*gtgt|thuế\s*gtgt|vat)\s*[:：]?\s*([\d.,]+)", text, re.I)
    subtotal = re.findall(r"(?:tiền trước thuế|cộng tiền hàng|thành tiền|subtotal)\s*[:：]?\s*([\d.,]+)", text, re.I)

    def num(vals):
        return int(re.sub(r"\D", "", vals[-1])) if vals else 0

    amount_val = num(amounts)
    vat_val = num(vat)
    sub_val = num(subtotal)

    if amount_val and sub_val and not vat_val and amount_val >= sub_val:
        vat_val = amount_val - sub_val
    elif sub_val and vat_val and not amount_val:
        amount_val = sub_val + vat_val
    elif amount_val and vat_val and not sub_val and amount_val >= vat_val:
        sub_val = amount_val - vat_val

    result["amount_vnd"] = amount_val
    result["vat_amount"] = vat_val
    result["subtotal_vnd"] = sub_val
    return result


def parse_invoice_with_ai_vision(uploaded_file, api_key: str, model: str = "gpt-4o-mini") -> dict:
    """Use OpenAI Vision to extract structured invoice fields from an uploaded image (JPG/PNG)."""
    raw_bytes = uploaded_file.getvalue()
    ext = uploaded_file.name.lower().rsplit(".", 1)[-1]
    mime = "image/png" if ext == "png" else "image/jpeg"
    b64_data = base64.b64encode(raw_bytes).decode("utf-8")

    client = OpenAI(api_key=api_key)
    prompt = (
        "Trích xuất thông tin hóa đơn từ ảnh này và trả về đúng một JSON object với các trường: "
        '{"invoice_id": "...", "tax_id": "...", "customer_name": "...", "issue_date": "YYYY-MM-DD", '
        '"subtotal_vnd": 0, "vat_amount": 0, "amount_vnd": 0, "raw_text": "..."}. '
        "Các trường tiền tệ phải là số nguyên VND. Chỉ trả về JSON, không dùng markdown."
    )
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": f"data:{mime};base64,{b64_data}"},
                ],
            }
        ],
        store=False,
    )
    out = response.output_text.strip()
    out = re.sub(r"^```(?:json)?\s*|\s*```$", "", out, flags=re.I).strip()
    data = json.loads(out)
    if "issue_date" in data and data["issue_date"]:
        data["issue_date"] = _normalize_date(str(data["issue_date"]))
    for k in ("subtotal_vnd", "vat_amount", "amount_vnd"):
        data[k] = int(data.get(k) or 0)
    return data
