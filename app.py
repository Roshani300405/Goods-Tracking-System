# ============================================================
# INVOICE INFORMATION EXTRACTOR — single-file Streamlit app
#
# ONE file, nothing else to create. On first run it auto-installs any
# missing Python packages (pip) so you don't need a separate
# requirements.txt.
#
# Run with:
#   streamlit run app.py
#
# IMPORTANT — one thing pip cannot install for you:
#   PDF support needs the "poppler" system binary (not a Python
#   package). This script will tell you clearly if it's missing and
#   how to install it, rather than failing with a cryptic error.
#     Debian/Ubuntu:            sudo apt-get install -y poppler-utils
#     macOS (Homebrew):         brew install poppler
#     Streamlit Community Cloud: add a packages.txt file containing
#                                 the single line: poppler-utils
#   Image-only invoices (png/jpg) work fine without poppler.
# ============================================================

import subprocess
import sys
import importlib

# ---------- SELF-INSTALLING PYTHON DEPENDENCIES ----------
_REQUIRED = {
    "streamlit": "streamlit",
    "cv2": "opencv-python-headless",
    "numpy": "numpy",
    "pandas": "pandas",
    "PIL": "pillow",
    "pdf2image": "pdf2image",
    "openpyxl": "openpyxl",
    "easyocr": "easyocr",
}

def _ensure_installed():
    missing = []
    for module_name, pip_name in _REQUIRED.items():
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(pip_name)
    if missing:
        print(f"Installing missing packages: {', '.join(missing)} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])

_ensure_installed()

import os
import io
import re
import json
import shutil

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from pdf2image import convert_from_bytes

POPPLER_AVAILABLE = shutil.which("pdftoppm") is not None

st.set_page_config(page_title="Invoice Information Extractor", page_icon="📄", layout="wide")

# ---------- OCR MODEL (loaded once, cached across reruns) ----------
@st.cache_resource(show_spinner="Loading OCR model (first run only)...")
def get_reader():
    import easyocr
    return easyocr.Reader(['en'], gpu=False)  # set gpu=True only if a CUDA GPU is available on the host

# ---------- LOAD PAGES (PDF or image bytes -> list of cv2 images) ----------
def load_pages_from_bytes(name, raw_bytes):
    ext = os.path.splitext(name)[1].lower()
    pages = []
    if ext == ".pdf":
        if not POPPLER_AVAILABLE:
            raise RuntimeError(
                "PDF support needs 'poppler' installed on this machine (not a Python "
                "package). Install it with 'sudo apt-get install -y poppler-utils' "
                "(Linux), 'brew install poppler' (Mac), or add a packages.txt file "
                "with 'poppler-utils' in it if deploying on Streamlit Community Cloud. "
                "Image files (png/jpg) work without it."
            )
        for p in convert_from_bytes(raw_bytes, dpi=300):
            pages.append(cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR))
    else:
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        pages.append(cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR))
    return pages

# ---------- LIGHT PREPROCESS (upscale + denoise, NOT hard binarize) ----------
def preprocess_image(img):
    h, w = img.shape[:2]
    if max(h, w) < 1800:
        scale = 1800 / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    return gray

# ---------- OCR WITH LINE RECONSTRUCTION ----------
def group_into_lines(results, y_threshold_ratio=0.6):
    if not results:
        return []
    items = []
    heights = []
    for bbox, text, conf in results:
        ys = [p[1] for p in bbox]
        xs = [p[0] for p in bbox]
        items.append({"y": sum(ys) / 4, "x": min(xs), "text": text})
        heights.append(max(ys) - min(ys))
    avg_h = (sum(heights) / len(heights)) if heights else 15
    y_thresh = avg_h * y_threshold_ratio

    items.sort(key=lambda i: i["y"])
    lines, current, current_y = [], [], None
    for it in items:
        if current_y is None or abs(it["y"] - current_y) <= y_thresh:
            current.append(it)
            current_y = it["y"] if current_y is None else (current_y + it["y"]) / 2
        else:
            current.sort(key=lambda i: i["x"])
            lines.append(current)
            current, current_y = [it], it["y"]
    if current:
        current.sort(key=lambda i: i["x"])
        lines.append(current)

    return [" ".join(i["text"] for i in ln) for ln in lines]

def run_ocr_lines(reader, img):
    processed = preprocess_image(img)
    results = reader.readtext(processed, detail=1, paragraph=False)
    return group_into_lines(results)

# ---------- NUMBER CLEANUP ----------
def clean_amount(raw):
    if not raw:
        return None
    raw = raw.replace(",", "").replace(" ", "")
    raw = re.sub(r"[^\d.]", "", raw)
    if raw.count(".") > 1:
        parts = raw.split(".")
        raw = "".join(parts[:-1]) + "." + parts[-1]
    try:
        return float(raw) if raw else None
    except ValueError:
        return None

# ---------- FIELD PATTERNS (broad, header-variant tolerant) ----------
FIELD_PATTERNS = {
    "invoice_number": r"(?:invoice|receipt|bill|ref(?:erence)?)\s*(?:no\.?|number|#|id)\s*[:\-]?\s*([A-Za-z0-9\-\/]{3,})",
    "order_id":       r"(?:order)\s*(?:no\.?|id|number)\s*[:\-]?\s*([A-Za-z0-9\-\/]{2,})",
    "invoice_date":   r"(?:invoice\s*date|bill\s*date|receipt\s*date|date\s*of\s*issue|^date)\s*[:\-]?\s*(\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4})",
    "due_date":       r"(?:due\s*date|payment\s*due)\s*[:\-]?\s*(\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4})",
    "gstin":          r"\b([0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[0-9A-Z]{1}[Z]{1}[0-9A-Z]{1})\b",
    "subtotal":       r"(?:sub\s*-?\s*total)\s*[:\-]?\s*[₹$]?\s*(Rs\.?)?\s*([\d,]+\.?\d*)",
    # NOTE: total_amount is handled separately below (needs a subtotal-exclusion
    # lookbehind + a fallback search order), not via a single regex here.
}

# Tried in order — first hit wins. The bare "total" alt has a negative
# lookbehind so it can NEVER match inside "Sub Total"/"Subtotal".
TOTAL_AMOUNT_PATTERNS = [
    r"(?:grand\s*total|net\s*(?:payable|amount)|amount\s*due|total\s*amount|invoice\s*total|balance\s*due)\s*[:\-]?\s*[₹$]?\s*(Rs\.?)?\s*([\d,]+\.?\d*)",
    r"(?<!sub)(?<!sub-)(?<!sub\s)\btotal\b\s*[:\-]?\s*[₹$]?\s*(Rs\.?)?\s*([\d,]+\.?\d*)",
]

# Tax lines are often formatted as "CGST @9% 72.00" or "SGST 9% : Rs. 72.00".
# A plain regex grabs whichever number comes right after the keyword — which
# is frequently the RATE (9), not the AMOUNT (72.00). So tax extraction is
# done per-line below: find lines mentioning a tax keyword, strip out any
# "<number>%" (the rate) first, then take the last remaining number on that
# line as the actual tax amount. Split-tax invoices (CGST + SGST) get summed.
TAX_KEYWORD_RE = re.compile(r"\b(?:cgst|sgst|igst|gst|vat|tax)\b", re.IGNORECASE)
PERCENT_NUMBER_RE = re.compile(r"[\d,]+\.?\d*\s*%")
AMOUNT_TOKEN_RE = re.compile(r"[\d,]+\.\d{1,2}|[\d,]{2,}")

def extract_tax_amount(lines):
    total_tax = 0.0
    found = False
    for ln in lines:
        if not TAX_KEYWORD_RE.search(ln):
            continue
        if re.search(r"\bgstin\b", ln, re.IGNORECASE):
            continue  # GSTIN numbers aren't tax amounts
        # Drop rate expressions like "9%", "18 %" so they can't be mistaken
        # for the amount.
        stripped = PERCENT_NUMBER_RE.sub("", ln)
        amounts = [clean_amount(tok) for tok in AMOUNT_TOKEN_RE.findall(stripped)]
        amounts = [a for a in amounts if a is not None]
        if amounts:
            # Amount is virtually always the last number on the line
            # (label, then rate already stripped, then amount).
            total_tax += amounts[-1]
            found = True
    return round(total_tax, 2) if found else None

SECTION_KEYWORDS = r"(?:bill\s*to|ship\s*to|invoice|gstin|total|date|subtotal|tax|hsn|qty|amount|item|description)"

SECTION_PATTERNS = {
    "buyer_name": rf"bill\s*to\s*[:\-]?\s*(.*?)(?=\n\s*(?:{SECTION_KEYWORDS})|\n\n|$)",
    "ship_to":    rf"ship\s*to\s*[:\-]?\s*(.*?)(?=\n\s*(?:{SECTION_KEYWORDS})|\n\n|$)",
}

LINE_ITEM_SKIP_WORDS = re.compile(
    r"\b(total|subtotal|sub-total|tax|gst|cgst|sgst|igst|vat|bill\s*to|ship\s*to|"
    r"invoice|gstin|date|amount\s*due|balance|discount|qty|description|hsn|s\.?no)\b",
    re.IGNORECASE,
)
LINE_ITEM_PATTERN = re.compile(r"^(.{3,80}?)[\s:]+[₹$]?\s*(Rs\.?)?\s*([\d,]+\.\d{1,2}|[\d,]{3,})\s*$")

def extract_line_items(lines):
    items = []
    for ln in lines:
        ln = ln.strip()
        if not ln or LINE_ITEM_SKIP_WORDS.search(ln):
            continue
        m = LINE_ITEM_PATTERN.match(ln)
        if m:
            desc = m.group(1).strip(" -:|")
            amount = clean_amount(m.group(3))
            if amount is not None and len(desc) >= 2:
                items.append({"description": desc, "amount": amount})
    return items

def guess_organization_name(lines):
    keyword_re = re.compile(r"(invoice|receipt|gstin|date|bill\s*to|ship\s*to|tax|order)", re.IGNORECASE)
    for ln in lines[:5]:
        ln = ln.strip()
        if len(ln) >= 3 and not keyword_re.search(ln) and not re.fullmatch(r"[\d\s.,\-\/]+", ln):
            return ln
    return None

def extract_fields(lines):
    text = "\n".join(lines)
    data = {}

    for field, pattern in FIELD_PATTERNS.items():
        m = re.search(pattern, text, re.IGNORECASE)
        if not m:
            data[field] = None
        elif field == "subtotal":
            data[field] = clean_amount(m.group(2))
        else:
            data[field] = m.group(1).strip()

    # ---- Tax amount: sum every tax line found (CGST + SGST + IGST etc.),
    # ignoring rate percentages like "9%" so only the actual amount is added.
    data["tax_amount"] = extract_tax_amount(lines)

    # ---- Total amount: try specific keywords first (grand total, amount
    # due, balance due, ...), then a bare "total" that can never match
    # inside "Subtotal" thanks to the negative lookbehind.
    total_amount = None
    for pattern in TOTAL_AMOUNT_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            total_amount = clean_amount(m.group(2))
            if total_amount is not None:
                break
    data["total_amount"] = total_amount

    # ---- Reconciliation fallback: if the OCR'd total is missing, or is
    # suspiciously <= subtotal (a sign the regex actually grabbed the
    # subtotal / only part of the tax), recompute it as subtotal + tax.
    data["total_amount_source"] = "ocr"
    if data.get("subtotal") is not None and data.get("tax_amount") is not None:
        computed_total = round(data["subtotal"] + data["tax_amount"], 2)
        if data["total_amount"] is None or data["total_amount"] <= data["subtotal"]:
            data["total_amount"] = computed_total
            data["total_amount_source"] = "computed (subtotal + tax)"

    for field, pattern in SECTION_PATTERNS.items():
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        data[field] = re.sub(r"\s+", " ", m.group(1)).strip()[:150] if m else None

    data["seller_name"] = guess_organization_name(lines)
    data["line_items"] = extract_line_items(lines)
    return data

# ---------- FULL PIPELINE FOR ONE FILE ----------
def process_invoice_bytes(reader, name, raw_bytes):
    pages = load_pages_from_bytes(name, raw_bytes)
    all_lines = []
    for page_img in pages:
        all_lines.extend(run_ocr_lines(reader, page_img))
    fields = extract_fields(all_lines)
    fields["source_file"] = name
    fields["raw_text"] = "\n".join(all_lines)
    return fields

SUMMARY_COLS = [
    "source_file", "invoice_number", "invoice_date", "due_date",
    "seller_name", "buyer_name", "ship_to", "gstin",
    "subtotal", "tax_amount", "total_amount", "total_amount_source",
]

# ============================================================
# STREAMLIT UI
# ============================================================

st.title("📄 Invoice Information Extractor")
st.caption("Upload one or more invoice images or PDFs, then click **Process Invoices**.")

if not POPPLER_AVAILABLE:
    st.warning(
        "PDF support is unavailable because 'poppler' isn't installed on this machine. "
        "Image files (png/jpg/tiff/bmp) still work fine. To enable PDFs: "
        "`sudo apt-get install -y poppler-utils` (Linux/Streamlit Cloud via packages.txt) "
        "or `brew install poppler` (Mac)."
    )

if "all_results" not in st.session_state:
    st.session_state.all_results = []
    st.session_state.summary_df = None
    st.session_state.items_df = None

uploaded_files = st.file_uploader(
    "Choose invoice files",
    type=["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp"],
    accept_multiple_files=True,
)

col1, col2 = st.columns([1, 1])
process_clicked = col1.button("Process Invoices", type="primary", disabled=not uploaded_files)
clear_clicked = col2.button("Clear / New Batch")

if clear_clicked:
    st.session_state.all_results = []
    st.session_state.summary_df = None
    st.session_state.items_df = None
    st.rerun()

if process_clicked and uploaded_files:
    reader = get_reader()

    progress_bar = st.progress(0, text="Starting...")
    status_box = st.empty()
    log_lines = []

    all_results = []
    for i, uf in enumerate(uploaded_files):
        raw_bytes = uf.getvalue()
        log_lines.append(f"Processing: {uf.name}")
        status_box.text("\n".join(log_lines))
        try:
            result = process_invoice_bytes(reader, uf.name, raw_bytes)
            all_results.append(result)
            log_lines.append(f"  done: {uf.name}")
        except Exception as e:
            log_lines.append(f"  failed on {uf.name}: {e}")
        status_box.text("\n".join(log_lines))
        progress_bar.progress((i + 1) / len(uploaded_files), text=f"{i + 1}/{len(uploaded_files)} processed")

    summary_rows = [{c: r.get(c) for c in SUMMARY_COLS} for r in all_results]
    summary_df = pd.DataFrame(summary_rows, columns=SUMMARY_COLS)

    item_rows = []
    for r in all_results:
        for item in r.get("line_items", []):
            item_rows.append({
                "source_file": r["source_file"],
                "description": item["description"],
                "amount": item["amount"],
            })
    items_df = pd.DataFrame(item_rows, columns=["source_file", "description", "amount"])

    st.session_state.all_results = all_results
    st.session_state.summary_df = summary_df
    st.session_state.items_df = items_df

    st.success("All files processed.")

# ---------- RESULTS + DOWNLOADS ----------
if st.session_state.summary_df is not None:
    st.subheader("Summary")
    st.dataframe(st.session_state.summary_df, use_container_width=True)

    st.subheader("Individual Line Items")
    if st.session_state.items_df.empty:
        st.info("No line items detected.")
    else:
        st.dataframe(st.session_state.items_df, use_container_width=True)

    dl_col1, dl_col2 = st.columns(2)

    json_bytes = json.dumps(st.session_state.all_results, indent=2, ensure_ascii=False).encode("utf-8")
    dl_col1.download_button(
        "Download JSON",
        data=json_bytes,
        file_name="extracted_invoices.json",
        mime="application/json",
    )

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        st.session_state.summary_df.to_excel(writer, sheet_name="Summary", index=False)
        st.session_state.items_df.to_excel(writer, sheet_name="Line Items", index=False)
    dl_col2.download_button(
        "Download Excel",
        data=excel_buffer.getvalue(),
        file_name="extracted_invoices.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
