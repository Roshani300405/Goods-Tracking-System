# Goods-Tracking-System
# 📄 Invoice Information Extractor

An AI-powered Invoice Information Extractor built using **Python**, **EasyOCR**, and **Google Colab**. This application automatically extracts key information from invoice images and PDF files, including invoice details and line items, and exports the results to JSON and Excel.

---

## 🚀 Features

- Upload one or more invoice images or PDF files
- Automatic OCR using EasyOCR
- Supports multiple invoice formats
- Extracts:
  - Invoice Number
  - Invoice Date
  - Due Date
  - Seller Name
  - Buyer Name
  - GSTIN
  - Subtotal
  - Tax Amount
  - Total Amount
- Detects invoice line items
- Batch processing of multiple invoices
- Download results as:
  - JSON
  - Excel (.xlsx)
- Interactive user interface using ipywidgets

---

## 🛠️ Technologies Used

- Python
- Google Colab
- EasyOCR
- OpenCV
- Pandas
- NumPy
- PDF2Image
- OpenPyXL
- ipywidgets
- PIL (Pillow)

---

## 📂 Project Structure

```
Invoice-Information-Extractor/
│
├── Invoice_Information_Extractor.ipynb
├── README.md
├── requirements.txt
├── sample_invoices/
├── extracted_invoices.json
├── extracted_invoices.xlsx
└── screenshots/
```

---

## 📥 Installation

Clone the repository

```bash
git clone https://github.com/your-username/Invoice-Information-Extractor.git
```

Move into the project folder

```bash
cd Invoice-Information-Extractor
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

1. Open the notebook in **Google Colab**.
2. Run all cells.
3. Upload one or more invoice files.
4. Click **Process Invoices**.
5. View extracted information.
6. Download the results as JSON or Excel.

---

## 📊 Output

The application extracts:

- Invoice Number
- Invoice Date
- Due Date
- Seller Name
- Buyer Name
- Ship To Address
- GSTIN
- Subtotal
- Tax Amount
- Total Amount
- Line Item Description
- Line Item Amount

---

## 📸 Screenshots

Add screenshots of:

- Upload Interface
- Processing
- Summary Table
- Extracted Line Items
- Download Buttons

Store them inside the `screenshots/` folder.

---

## 🎯 Future Improvements

- Support handwritten invoices
- Improve table extraction
- Add multilingual OCR
- Deploy as a web application
- Cloud storage integration

---

## 👩‍💻 Author

**Vadarevu Roshani**

B.Tech – Artificial Intelligence and Data Science

---

## ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub.
