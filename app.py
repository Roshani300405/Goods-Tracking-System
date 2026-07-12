# ============================================================
# INVOICE INFORMATION EXTRACTOR — v3, SINGLE CELL, RUN ONCE
# Paste this entire block into ONE Google Colab cell and run it.
# Setup (installs + OCR model load) happens ONCE.
#
# WHAT'S NEW IN v3 (frontend added):
#  - A proper in-notebook UI built with ipywidgets: an "Upload Invoices"
#    button, a live progress bar, a status log, and a results area that
#    renders the Summary + Line Items tables directly in the output —
#    no more scrolling through print()/display() calls.
#  - Two download buttons ("Download JSON" / "Download Excel") that only
#    appear once processing finishes, so you can re-run a batch without
#    triggering a download every time.
#  - A "Clear / New Batch" button to reset the UI and process another
#    set of invoices without re-running the whole cell.
#  - All backend logic (OCR, line reconstruction, field extraction,
#    line-item extraction) is UNCHANGED from v2 — only the interaction
#    layer around it is new.
# ============================================================

# ---------- 1. INSTALL (runs once) ----------
!pip install -q easyocr opencv-python-headless pdf2image openpyxl ipywidgets
!apt-get -qq install -y poppler-utils > /dev/null

# ---------- 2. IMPORTS ----------
import cv2
import re
import json
import io
import numpy as np
import pandas as pd
import easyocr
from pdf2image import convert_from_path
from google.colab import files
from PIL import Image
import os

import ipywidgets as widgets
from IPython.display import display, clear_output, HTML

# ---------- 3. LOAD OCR MODEL (runs once) ----------
print("Loading OCR model... (this happens only once)")
reader = easyocr.Reader(['en'], gpu=True)
print("OCR model ready.\n")

# ---------- 4. LOAD PAGES (PDF or image -> list of cv2 images) ----------
def load_pages_from_bytes(name, raw_bytes):
    ext = os.path.splitext(name)[1].lower()
    pages = []
    if ext == ".pdf":
        tmp_path = f"/tmp/{name}"
        with open(tmp_path, "wb") as f:
            f.write(raw_bytes)
        for p in convert_from_path(tmp_path, dpi=300):
            pages.append(cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR))
    else:
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        pages.append(cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR))
    return pages

# ---------- 5. LIGHT PREPROCESS (upscale + denoise, NOT hard binarize) ----------
def preprocess_image(img):
    h, w = img.shape[:2]
    if max(h, w) < 1800:
        scale = 1800 / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    return gray

# ---------- 6. OCR WITH LINE RECONSTRUCTION ----------
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

def run_ocr_lines(img):
    processed = preprocess_image(img)
    results = reader.readtext(processed, detail=1, paragraph=False)
    return group_into_lines(results)

# ---------- 7. NUMBER CLEANUP ----------
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

# ---------- 8. FIELD PATTERNS (broad, header-variant tolerant) ----------
FIELD_PATTERNS = {
    "invoice_number": r"(?:invoice|receipt|bill|ref(?:erence)?)\s*(?:no\.?|number|#|id)\s*[:\-]?\s*([A-Za-z0-9\-\/]{3,})",
    "order_id":       r"(?:order)\s*(?:no\.?|id|number)\s*[:\-]?\s*([A-Za-z0-9\-\/]{2,})",
    "invoice_date":   r"(?:invoice\s*date|bill\s*date|receipt\s*date|date\s*of\s*issue|^date)\s*[:\-]?\s*(\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4})",
    "due_date":       r"(?:due\s*date|payment\s*due)\s*[:\-]?\s*(\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4})",
    "gstin":          r"\b([0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[0-9A-Z]{1}[Z]{1}[0-9A-Z]{1})\b",
    "subtotal":       r"(?:sub\s*-?\s*total)\s*[:\-]?\s*[₹$]?\s*(Rs\.?)?\s*([\d,]+\.?\d*)",
    "tax_amount":     r"(?:(?:cgst|sgst|igst|gst|vat|tax)\s*(?:amount)?)\s*[:\-]?\s*[₹$]?\s*(Rs\.?)?\s*([\d,]+\.?\d*)",
    "total_amount":   r"(?:grand\s*total|net\s*(?:payable|amount)|amount\s*due|total\s*amount|invoice\s*total|balance\s*due|total)\s*[:\-]?\s*[₹$]?\s*(Rs\.?)?\s*([\d,]+\.?\d*)",
}

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
        elif field in ("subtotal", "tax_amount", "total_amount"):
            data[field] = clean_amount(m.group(2))
        else:
            data[field] = m.group(1).strip()

    for field, pattern in SECTION_PATTERNS.items():
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        data[field] = re.sub(r"\s+", " ", m.group(1)).strip()[:150] if m else None

    data["seller_name"] = guess_organization_name(lines)
    data["line_items"] = extract_line_items(lines)
    return data

# ---------- 9. FULL PIPELINE FOR ONE FILE (works from raw bytes now) ----------
def process_invoice_bytes(name, raw_bytes):
    pages = load_pages_from_bytes(name, raw_bytes)
    all_lines = []
    for page_img in pages:
        all_lines.extend(run_ocr_lines(page_img))
    fields = extract_fields(all_lines)
    fields["source_file"] = name
    fields["raw_text"] = "\n".join(all_lines)
    return fields

SUMMARY_COLS = [
    "source_file", "invoice_number", "invoice_date", "due_date",
    "seller_name", "buyer_name", "ship_to", "gstin",
    "subtotal", "tax_amount", "total_amount",
]

# ============================================================
# 10. FRONTEND — ipywidgets UI
# ============================================================

upload_widget = widgets.FileUpload(
    accept=".pdf,.png,.jpg,.jpeg,.tif,.tiff,.bmp",
    multiple=True,
    description="Choose files",
    button_style="",
    layout=widgets.Layout(width="220px"),
)
process_btn = widgets.Button(
    description="Process Invoices",
    button_style="primary",
    icon="cogs",
    disabled=True,
    layout=widgets.Layout(width="220px"),
)
clear_btn = widgets.Button(
    description="Clear / New Batch",
    button_style="warning",
    icon="refresh",
    layout=widgets.Layout(width="220px"),
)
download_json_btn = widgets.Button(
    description="Download JSON",
    button_style="success",
    icon="download",
    disabled=True,
    layout=widgets.Layout(width="220px"),
)
download_xlsx_btn = widgets.Button(
    description="Download Excel",
    button_style="success",
    icon="download",
    disabled=True,
    layout=widgets.Layout(width="220px"),
)

progress = widgets.IntProgress(
    value=0, min=0, max=1, description="Idle:",
    layout=widgets.Layout(width="500px"),
)
status_log = widgets.Output(layout=widgets.Layout(
    border="1px solid #ddd", padding="6px", height="120px", overflow="auto"
))
results_out = widgets.Output()

title_html = widgets.HTML(
    "<h2 style='margin-bottom:0'>📄 Invoice Information Extractor</h2>"
    "<p style='color:#666;margin-top:4px'>Upload one or more invoice images or PDFs, "
    "then click <b>Process Invoices</b>.</p>"
)

top_buttons = widgets.HBox([upload_widget, process_btn, clear_btn])
download_buttons = widgets.HBox([download_json_btn, download_xlsx_btn])

ui = widgets.VBox([
    title_html,
    top_buttons,
    progress,
    status_log,
    results_out,
    download_buttons,
])

# state kept across button clicks
_state = {"all_results": [], "summary_df": None, "items_df": None}

def log(msg):
    with status_log:
        print(msg)

def _on_upload_change(change):
    process_btn.disabled = len(upload_widget.value) == 0

upload_widget.observe(_on_upload_change, names="value")

def _render_results(summary_df, items_df):
    with results_out:
        clear_output(wait=True)
        display(HTML("<h3>Summary</h3>"))
        display(summary_df)
        display(HTML("<h3>Individual Line Items</h3>"))
        if items_df.empty:
            display(HTML("<p style='color:#999'>No line items detected.</p>"))
        else:
            display(items_df)

def _on_process_clicked(b):
    uploaded_files = upload_widget.value  # tuple of dicts (name, content, ...) in recent ipywidgets
    if not uploaded_files:
        return

    process_btn.disabled = True
    upload_widget.disabled = True
    download_json_btn.disabled = True
    download_xlsx_btn.disabled = True

    with status_log:
        clear_output()
    with results_out:
        clear_output()

    # Normalize FileUpload.value across ipywidgets versions (dict-of-dicts vs tuple-of-dicts)
    if isinstance(uploaded_files, dict):
        file_items = [(name, meta["content"]) for name, meta in uploaded_files.items()]
    else:
        file_items = [(meta["name"], meta["content"]) for meta in uploaded_files]

    progress.max = len(file_items)
    progress.value = 0
    progress.description = "Working:"

    all_results = []
    for name, content in file_items:
        raw_bytes = bytes(content)
        log(f"Processing: {name}")
        try:
            result = process_invoice_bytes(name, raw_bytes)
            all_results.append(result)
            log(f"  ✓ done: {name}")
        except Exception as e:
            log(f"  ✗ failed on {name}: {e}")
        progress.value += 1

    progress.description = "Done:"

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

    _state["all_results"] = all_results
    _state["summary_df"] = summary_df
    _state["items_df"] = items_df

    _render_results(summary_df, items_df)

    download_json_btn.disabled = False
    download_xlsx_btn.disabled = False
    log("\nAll files processed. Use the download buttons below to save results.")

def _on_download_json(b):
    if not _state["all_results"]:
        return
    with open("extracted_invoices.json", "w", encoding="utf-8") as f:
        json.dump(_state["all_results"], f, indent=2, ensure_ascii=False)
    files.download("extracted_invoices.json")

def _on_download_xlsx(b):
    if _state["summary_df"] is None:
        return
    with pd.ExcelWriter("extracted_invoices.xlsx", engine="openpyxl") as writer:
        _state["summary_df"].to_excel(writer, sheet_name="Summary", index=False)
        _state["items_df"].to_excel(writer, sheet_name="Line Items", index=False)
    files.download("extracted_invoices.xlsx")

def _on_clear_clicked(b):
    _state["all_results"] = []
    _state["summary_df"] = None
    _state["items_df"] = None
    upload_widget.value.clear() if isinstance(upload_widget.value, dict) else None
    upload_widget._counter = 0
    upload_widget.value = () if not isinstance(upload_widget.value, dict) else {}
    upload_widget.disabled = False
    process_btn.disabled = True
    download_json_btn.disabled = True
    download_xlsx_btn.disabled = True
    progress.value = 0
    progress.max = 1
    progress.description = "Idle:"
    with status_log:
        clear_output()
    with results_out:
        clear_output()

process_btn.on_click(_on_process_clicked)
download_json_btn.on_click(_on_download_json)
download_xlsx_btn.on_click(_on_download_xlsx)
clear_btn.on_click(_on_clear_clicked)

display(ui)# ============================================================
# INVOICE INFORMATION EXTRACTOR — v3, SINGLE CELL, RUN ONCE
# Paste this entire block into ONE Google Colab cell and run it.
# Setup (installs + OCR model load) happens ONCE.
#
# WHAT'S NEW IN v3 (frontend added):
#  - A proper in-notebook UI built with ipywidgets: an "Upload Invoices"
#    button, a live progress bar, a status log, and a results area that
#    renders the Summary + Line Items tables directly in the output —
#    no more scrolling through print()/display() calls.
#  - Two download buttons ("Download JSON" / "Download Excel") that only
#    appear once processing finishes, so you can re-run a batch without
#    triggering a download every time.
#  - A "Clear / New Batch" button to reset the UI and process another
#    set of invoices without re-running the whole cell.
#  - All backend logic (OCR, line reconstruction, field extraction,
#    line-item extraction) is UNCHANGED from v2 — only the interaction
#    layer around it is new.
# ============================================================

# ---------- 1. INSTALL (runs once) ----------
!pip install -q easyocr opencv-python-headless pdf2image openpyxl ipywidgets
!apt-get -qq install -y poppler-utils > /dev/null

# ---------- 2. IMPORTS ----------
import cv2
import re
import json
import io
import numpy as np
import pandas as pd
import easyocr
from pdf2image import convert_from_path
from google.colab import files
from PIL import Image
import os

import ipywidgets as widgets
from IPython.display import display, clear_output, HTML

# ---------- 3. LOAD OCR MODEL (runs once) ----------
print("Loading OCR model... (this happens only once)")
reader = easyocr.Reader(['en'], gpu=True)
print("OCR model ready.\n")

# ---------- 4. LOAD PAGES (PDF or image -> list of cv2 images) ----------
def load_pages_from_bytes(name, raw_bytes):
    ext = os.path.splitext(name)[1].lower()
    pages = []
    if ext == ".pdf":
        tmp_path = f"/tmp/{name}"
        with open(tmp_path, "wb") as f:
            f.write(raw_bytes)
        for p in convert_from_path(tmp_path, dpi=300):
            pages.append(cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR))
    else:
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        pages.append(cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR))
    return pages

# ---------- 5. LIGHT PREPROCESS (upscale + denoise, NOT hard binarize) ----------
def preprocess_image(img):
    h, w = img.shape[:2]
    if max(h, w) < 1800:
        scale = 1800 / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    return gray

# ---------- 6. OCR WITH LINE RECONSTRUCTION ----------
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

def run_ocr_lines(img):
    processed = preprocess_image(img)
    results = reader.readtext(processed, detail=1, paragraph=False)
    return group_into_lines(results)

# ---------- 7. NUMBER CLEANUP ----------
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

# ---------- 8. FIELD PATTERNS (broad, header-variant tolerant) ----------
FIELD_PATTERNS = {
    "invoice_number": r"(?:invoice|receipt|bill|ref(?:erence)?)\s*(?:no\.?|number|#|id)\s*[:\-]?\s*([A-Za-z0-9\-\/]{3,})",
    "order_id":       r"(?:order)\s*(?:no\.?|id|number)\s*[:\-]?\s*([A-Za-z0-9\-\/]{2,})",
    "invoice_date":   r"(?:invoice\s*date|bill\s*date|receipt\s*date|date\s*of\s*issue|^date)\s*[:\-]?\s*(\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4})",
    "due_date":       r"(?:due\s*date|payment\s*due)\s*[:\-]?\s*(\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4})",
    "gstin":          r"\b([0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[0-9A-Z]{1}[Z]{1}[0-9A-Z]{1})\b",
    "subtotal":       r"(?:sub\s*-?\s*total)\s*[:\-]?\s*[₹$]?\s*(Rs\.?)?\s*([\d,]+\.?\d*)",
    "tax_amount":     r"(?:(?:cgst|sgst|igst|gst|vat|tax)\s*(?:amount)?)\s*[:\-]?\s*[₹$]?\s*(Rs\.?)?\s*([\d,]+\.?\d*)",
    "total_amount":   r"(?:grand\s*total|net\s*(?:payable|amount)|amount\s*due|total\s*amount|invoice\s*total|balance\s*due|total)\s*[:\-]?\s*[₹$]?\s*(Rs\.?)?\s*([\d,]+\.?\d*)",
}

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
        elif field in ("subtotal", "tax_amount", "total_amount"):
            data[field] = clean_amount(m.group(2))
        else:
            data[field] = m.group(1).strip()

    for field, pattern in SECTION_PATTERNS.items():
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        data[field] = re.sub(r"\s+", " ", m.group(1)).strip()[:150] if m else None

    data["seller_name"] = guess_organization_name(lines)
    data["line_items"] = extract_line_items(lines)
    return data

# ---------- 9. FULL PIPELINE FOR ONE FILE (works from raw bytes now) ----------
def process_invoice_bytes(name, raw_bytes):
    pages = load_pages_from_bytes(name, raw_bytes)
    all_lines = []
    for page_img in pages:
        all_lines.extend(run_ocr_lines(page_img))
    fields = extract_fields(all_lines)
    fields["source_file"] = name
    fields["raw_text"] = "\n".join(all_lines)
    return fields

SUMMARY_COLS = [
    "source_file", "invoice_number", "invoice_date", "due_date",
    "seller_name", "buyer_name", "ship_to", "gstin",
    "subtotal", "tax_amount", "total_amount",
]

# ============================================================
# 10. FRONTEND — ipywidgets UI
# ============================================================

upload_widget = widgets.FileUpload(
    accept=".pdf,.png,.jpg,.jpeg,.tif,.tiff,.bmp",
    multiple=True,
    description="Choose files",
    button_style="",
    layout=widgets.Layout(width="220px"),
)
process_btn = widgets.Button(
    description="Process Invoices",
    button_style="primary",
    icon="cogs",
    disabled=True,
    layout=widgets.Layout(width="220px"),
)
clear_btn = widgets.Button(
    description="Clear / New Batch",
    button_style="warning",
    icon="refresh",
    layout=widgets.Layout(width="220px"),
)
download_json_btn = widgets.Button(
    description="Download JSON",
    button_style="success",
    icon="download",
    disabled=True,
    layout=widgets.Layout(width="220px"),
)
download_xlsx_btn = widgets.Button(
    description="Download Excel",
    button_style="success",
    icon="download",
    disabled=True,
    layout=widgets.Layout(width="220px"),
)

progress = widgets.IntProgress(
    value=0, min=0, max=1, description="Idle:",
    layout=widgets.Layout(width="500px"),
)
status_log = widgets.Output(layout=widgets.Layout(
    border="1px solid #ddd", padding="6px", height="120px", overflow="auto"
))
results_out = widgets.Output()

title_html = widgets.HTML(
    "<h2 style='margin-bottom:0'>📄 Invoice Information Extractor</h2>"
    "<p style='color:#666;margin-top:4px'>Upload one or more invoice images or PDFs, "
    "then click <b>Process Invoices</b>.</p>"
)

top_buttons = widgets.HBox([upload_widget, process_btn, clear_btn])
download_buttons = widgets.HBox([download_json_btn, download_xlsx_btn])

ui = widgets.VBox([
    title_html,
    top_buttons,
    progress,
    status_log,
    results_out,
    download_buttons,
])

# state kept across button clicks
_state = {"all_results": [], "summary_df": None, "items_df": None}

def log(msg):
    with status_log:
        print(msg)

def _on_upload_change(change):
    process_btn.disabled = len(upload_widget.value) == 0

upload_widget.observe(_on_upload_change, names="value")

def _render_results(summary_df, items_df):
    with results_out:
        clear_output(wait=True)
        display(HTML("<h3>Summary</h3>"))
        display(summary_df)
        display(HTML("<h3>Individual Line Items</h3>"))
        if items_df.empty:
            display(HTML("<p style='color:#999'>No line items detected.</p>"))
        else:
            display(items_df)

def _on_process_clicked(b):
    uploaded_files = upload_widget.value  # tuple of dicts (name, content, ...) in recent ipywidgets
    if not uploaded_files:
        return

    process_btn.disabled = True
    upload_widget.disabled = True
    download_json_btn.disabled = True
    download_xlsx_btn.disabled = True

    with status_log:
        clear_output()
    with results_out:
        clear_output()

    # Normalize FileUpload.value across ipywidgets versions (dict-of-dicts vs tuple-of-dicts)
    if isinstance(uploaded_files, dict):
        file_items = [(name, meta["content"]) for name, meta in uploaded_files.items()]
    else:
        file_items = [(meta["name"], meta["content"]) for meta in uploaded_files]

    progress.max = len(file_items)
    progress.value = 0
    progress.description = "Working:"

    all_results = []
    for name, content in file_items:
        raw_bytes = bytes(content)
        log(f"Processing: {name}")
        try:
            result = process_invoice_bytes(name, raw_bytes)
            all_results.append(result)
            log(f"  ✓ done: {name}")
        except Exception as e:
            log(f"  ✗ failed on {name}: {e}")
        progress.value += 1

    progress.description = "Done:"

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

    _state["all_results"] = all_results
    _state["summary_df"] = summary_df
    _state["items_df"] = items_df

    _render_results(summary_df, items_df)

    download_json_btn.disabled = False
    download_xlsx_btn.disabled = False
    log("\nAll files processed. Use the download buttons below to save results.")

def _on_download_json(b):
    if not _state["all_results"]:
        return
    with open("extracted_invoices.json", "w", encoding="utf-8") as f:
        json.dump(_state["all_results"], f, indent=2, ensure_ascii=False)
    files.download("extracted_invoices.json")

def _on_download_xlsx(b):
    if _state["summary_df"] is None:
        return
    with pd.ExcelWriter("extracted_invoices.xlsx", engine="openpyxl") as writer:
        _state["summary_df"].to_excel(writer, sheet_name="Summary", index=False)
        _state["items_df"].to_excel(writer, sheet_name="Line Items", index=False)
    files.download("extracted_invoices.xlsx")

def _on_clear_clicked(b):
    _state["all_results"] = []
    _state["summary_df"] = None
    _state["items_df"] = None
    upload_widget.value.clear() if isinstance(upload_widget.value, dict) else None
    upload_widget._counter = 0
    upload_widget.value = () if not isinstance(upload_widget.value, dict) else {}
    upload_widget.disabled = False
    process_btn.disabled = True
    download_json_btn.disabled = True
    download_xlsx_btn.disabled = True
    progress.value = 0
    progress.max = 1
    progress.description = "Idle:"
    with status_log:
        clear_output()
    with results_out:
        clear_output()

process_btn.on_click(_on_process_clicked)
download_json_btn.on_click(_on_download_json)
download_xlsx_btn.on_click(_on_download_xlsx)
clear_btn.on_click(_on_clear_clicked)

display(ui)
