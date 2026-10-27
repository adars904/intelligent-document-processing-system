# Document AI System — CNN Classification + OCR Extraction + Human Review

End-to-end document processing pipeline: classifies scanned documents into 5
types, extracts structured fields via OCR, and routes low-confidence results
to human review. Built as a placement portfolio project.

**Timeline:** Sept 20 – nov 10, 2026 (50days)

---

# Project phases

- [x] **Phase 1 — CNN baseline + dataset manifest** *(completed)*
- [x] **Phase 2 — Transfer learning (MobileNetV2)** *(completed)*
- [x] **Phase 2.5 — Modern data collection** *(completed)*
- [x] **Phase 2.6 — Combined retraining (old + modern)** *(completed)*
- [x] **Phase 3 — OCR** *(completed)*
- [ ] Phase 4 — Extraction
- [ ] Phase 5 — Validation / confidence scoring
- [ ] Phase 6 — FastAPI + SQLite + model versioning + audit log
- [ ] Phase 7 — React dashboard
- [ ] Phase 8 — Testing + golden set + adversarial fixtures + Known Failure Modes doc
- [ ] Phase 9 — Integration + docs + Future Work (async job queue)
---

## Dataset

### Original (Phase 1–2)

5 document classes, ~4,400 images total.

| Class | Source | Type | Count |
|---|---|---|---|
| Invoice | RVL-CDIP (Hugging Face) | Real | 800 |
| Form | RVL-CDIP | Real | 800 |
| Resume | RVL-CDIP | Real | 800 |
| ID Card | Custom synthetic generator | Synthetic | 1,400 |
| Certificate (10th/12th marksheet, degree) | Custom synthetic generator | Synthetic | 600 |

**Note on ID Card count:** this class contains 4 subtypes generated
independently — Aadhaar-style (~800), Employee ID (~200), Student ID (~200),
and Company ID (~200) — intentionally kept as a single `id_card` class rather
than split into subtypes. This makes it larger than the other 4 classes
(600–800 each). The resulting class imbalance is handled via `class_weight`
during CNN training rather than by downsampling the data.

### Modern data added in Phase 2.5

To address domain shift on modern documents (clean PDFs, digital resumes),
modern images were added for the three RVL-CDIP classes:

| Class | Original (RVL-CDIP) | Modern added | Total |
|---|---|---|---|
| Invoice | 800 | 450 | 1,250 |
| Form | 800 | 450 | 1,250 |
| Resume | 800 | 450 | 1,250 |
| ID Card | 1,400 (synthetic) | 0 | 1,400 |
| Certificate | 600 (synthetic) | 0 | 600 |
| **Total** | **4,400** | **1,350** | **5,750** |

**Modern sources:**
- **Resume:** Resumes-Images-Datasets (Kaggle, ~12.7k images) —
  150 each from Bing_images, Scrapped_Resumes, resume_database
- **Invoice:** High-Quality Invoice Images for OCR (Kaggle, ~8.2k images) —
  150 each from batch_1, batch_2, batch_3
- **Form:** FUNSD+ (Hugging Face, 1,026 images) — 450 sampled

**Rationale:** original training data was entirely historical microfilm scans
(RVL-CDIP) plus synthetic ID Card / Certificate images. Models trained only
on historical scans fail on modern clean documents due to domain shift.
Adding 450 modern images per class teaches the model to recognize document
structure independent of style (scan noise vs clean render).

### Folder structure (after Phase 2.6 merge)

```
dataset/
├── certificate/
├── form/
│   ├── (original RVL-CDIP images)
│   └── form_modern/         # 450 modern
├── id_card/
├── invoice/
│   ├── (original RVL-CDIP images)
│   └── invoice_modern/      # 450 modern
└── resume/
    ├── (original RVL-CDIP images)
    └── resume_modern/       # 450 modern
```

---

## Phase 1 status

### Task 1 — Dataset manifest ✅ Complete

`generate_manifest.py` walks the `dataset/` folder and produces `manifest.csv`,
the single source of truth for train/val/test split membership used by every
later phase.

- Infers `source` (real/synthetic) from folder name
- Hashes every file to flag exact-duplicate images (`duplicates.csv`)
- 70/15/15 train/val/test split, stratified jointly by `class_name` + `source` + `variant`
- Detects and removes cross-split duplicate leakage

**Run:**
```bash
python generate_manifest.py --dataset-root dataset --out manifest.csv
```

**Output columns:** `filepath, filename, class_name, source, variant, file_hash, split`

### Task 2 — CNN baseline (from scratch) ✅ Complete

Plain CNN, no pretrained weights, to establish a floor accuracy before
Phase 2's transfer-learning model. Reads `manifest.csv` directly so
train/val/test membership stays fixed across every phase.

---

## Phase 2 status — Transfer learning

MobileNetV2 transfer learning, 320×320 input, two-phase training:

- **Phase 2A:** frozen base, train classification head only (Adam, lr=1e-3)
- **Phase 2B:** unfreeze last 50 layers of base, fine-tune (Adam, lr=1e-5)

Data augmentation: horizontal flip, rotation, zoom, translation.

### Results (Phase 2, original data only)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| certificate | 1.00 | 1.00 | **1.00** |
| form | 0.65 | 0.88 | **0.75** |
| id_card | 1.00 | 1.00 | **1.00** |
| invoice | 0.86 | 0.57 | **0.68** |
| resume | 0.93 | 0.93 | **0.93** |
| **Overall** | | | **0.886** |

**Test accuracy: 88.6%**

### Known limitation identified in Phase 2

Invoice recall (0.57) and Form precision (0.65) are bounded by RVL-CDIP source
quality. The original RVL-CDIP Invoice and Form images are microfilm scans
from the 1960s–1990s whose text is unreadable at the pixel level for both
human annotators and the model.

**Resolution improvement (160→320) yielded +5.3%; deepening fine-tuning
(30→50 layers) yielded +1.5%. Diminishing returns indicate the remaining
error is a data limitation, not a model capacity limitation.**

---

## Phase 2.5 status — Modern data collection ✅ Complete

**1,350 modern images collected** across the three RVL-CDIP classes
(450 each for Invoice, Form, Resume).

| Class | Modern images | Sources |
|---|---|---|
| Resume | 450 | Kaggle Resumes-Images-Datasets (150 × 3 subfolders) |
| Invoice | 450 | Kaggle High-Quality Invoice Images (150 × 3 batches) |
| Form | 450 | Hugging Face FUNSD+ |
| **Total** | **1,350** | |

---

## Phase 2.6 status — Combined retraining ✅ Complete

Merged modern data into the original dataset and retrained MobileNetV2 with
identical Phase 2 hyperparameters, plus `class_weight` to handle the ID Card /
Certificate imbalance.

### Results (Phase 2.6, combined data)

**Full test set (858 images — old + modern mixed):**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| certificate | 1.00 | 1.00 | **1.00** |
| form | 0.75 | 0.77 | **0.76** |
| id_card | 1.00 | 1.00 | **1.00** |
| **invoice** | **0.82** | **0.77** | **0.79** ⬆️ |
| resume | 0.90 | 0.95 | **0.92** |
| **Overall** | | | **0.887** |

**Test accuracy: 88.69%**

**Modern-only test set (198 images — clean digital documents):**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| form | 0.98 | 0.76 | 0.86 |
| invoice | 0.85 | 0.98 | 0.91 |
| resume | 0.93 | 1.00 | 0.96 |
| **Overall** | | | **0.914** |

**Modern-only accuracy: 91.41%**

### Phase 2 → Phase 2.6 comparison

| Class | Phase 2 F1 | Phase 2.6 F1 | Change |
|---|---|---|---|
| certificate | 1.00 | 1.00 | — |
| form | 0.75 | 0.76 | +0.01 |
| id_card | 1.00 | 1.00 | — |
| **invoice** | **0.68** | **0.79** | **+0.11** ⬆️ |
| resume | 0.93 | 0.92 | — |
| **Overall** | 0.886 | 0.887 | +0.001 |

**Key win:** Invoice recall improved from **0.57 → 0.77** on the mixed test
set, and to **0.98** on modern-only data.

**Remaining limitation:** Form recall on modern data is 0.76 — some modern
forms are misclassified as invoices due to overlapping layout structure.

### Model artifacts

| Artifact | Location |
|---|---|
| Trained model | `transfer_model_phase25.keras` |
| Training history | `results/transfer_results_phase25.json` |
| Classification report | `results/classification_report_transfer_phase25.json` |
| Confusion matrix | `results/confusion_matrix_transfer_phase25.png` |

---  
## Phase 3 — OCR ✅ Complete

**Goal:** Convert document images and PDF pages into OCR text, inspect detected words and their positions/confidence, and prepare the text for downstream field extraction.

**Stack:** OpenCV → Tesseract (`pytesseract`) → `pdf2image` + Poppler for PDF conversion.

**Learning notebook:** `notebooks/phase3_ocr_learning.ipynb`

### What we learned and tested

| Area                              | Topics covered                                                                                                                                               |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| OpenCV image basics               | Image shape `(height, width, channels)`, pixels, RGB/BGR channels, loading and displaying images, grayscale conversion                                       |
| Image preprocessing               | Thresholding, adaptive thresholding, Gaussian blur basics, resizing, and comparing preprocessing results                                                     |
| Tesseract OCR                     | Installing/importing `pytesseract`, `image_to_string()`, understanding OCR output, and PSM 6, 7 and 8                                                        |
| OCR bounding boxes and confidence | `image_to_data()`, word text, `left`, `top`, `width`, `height`, confidence scores, looping through detected words, and drawing boxes using `cv2.rectangle()` |
| Blurry-document testing           | Comparing original, grayscale and adaptive-threshold images; testing resizing; understanding preprocessing limitations                                       |
| PDF-to-image conversion           | Using `pdf2image`, converting PDF pages to images, and passing a converted page to Tesseract                                                                 |
| OCR text cleaning                 | Removing surrounding whitespace, normalising whitespace, cleaning unwanted characters with regex, and understanding why line breaks matter for extraction    |

### Key concepts learned

- **`image_to_string()` vs `image_to_data()`:** `image_to_string()` returns recognised text. `image_to_data()` provides recognised text along with positional information and confidence values.
- **Confidence is a review signal, not proof of correctness.** Low-confidence text may indicate a problem, while high confidence does not guarantee that the recognised text is correct.
- **Bounding boxes locate text; they do not identify its business meaning.** Determining which number represents an invoice total belongs to Phase 4 — Extraction.
- **PSM affects text recognition.** Different page segmentation modes suit different layouts, such as a full page, a block of text, a line, a word or a character.
- **Preprocessing has limits.** Thresholding or resizing may help some documents, but severe blur can still produce unreadable output.
- **OCR text structure matters.** Removing all line breaks can make it harder to associate field labels with their values.

### Experiment 1 — Clean digital invoice

A clean digital invoice was processed using Tesseract. We inspected the recognised text and word-level confidence information.

This helped demonstrate that OCR can recognise clean printed content effectively, while decorative text and some table labels may still be misread.

### Experiment 2 — Blurry invoice

We compared OCR results from the original image, grayscale conversion, adaptive thresholding and resizing.

**Conclusion:** Preprocessing can help with some difficult images, but severe blur cannot reliably be repaired through preprocessing alone. Unreadable documents should be flagged for human review rather than trusted automatically.

### Experiment 3 — PDF invoice

A real multi-page invoice PDF was converted into page images using `pdf2image`. We then ran OCR on a converted page.

The OCR text contained invoice details and line items, but table columns were sometimes flattened or merged. This showed why extracting structured fields from invoice OCR requires additional rules and validation.

### Decisions carried forward

| Decision                                                                            | Relevant phase           |
| ----------------------------------------------------------------------------------- | ------------------------ |
| Use OCR text as the input to field extraction                                       | Phase 4                  |
| Use `image_to_data()` when word positions and OCR confidence are needed             | Phases 4–5               |
| Preserve useful line breaks and field labels during text cleaning                   | Phase 4                  |
| Convert PDF pages to images before running Tesseract                                | OCR pipeline integration |
| Flag empty or potentially unreliable OCR output for human review                    | Phases 5–7               |
| Test OCR configuration against a varied document set before choosing final settings | Testing and integration  |

### Known limitations

- Severely blurred or low-resolution images can produce empty or incorrect OCR text.
- Logos, decorative fonts and unusual layouts may be recognised incorrectly.
- Invoice tables may be flattened, causing columns and values to lose their original relationships.
- OCR recognition errors can affect downstream extraction.
- OCR confidence does not guarantee correctness.
- PDF conversion and OCR need to be handled page by page for multi-page documents.
- The experiments so far do not establish reliable OCR performance across every document type, language or scan quality.

### Next — Phase 4: Rule-Based Field Extraction

**Status: In progress.**

We have started applying regex rules to real invoice OCR text. The initial target fields include:

- Invoice number
- Invoice date and order date
- Seller name and GSTIN
- Customer details where available
- Subtotal, tax, shipping and total amount
- Email and phone number where available

The first rules are prototypes, not a fully validated extractor. The real invoice example demonstrated that OCR text can contain inconsistent labels, repeated values and scrambled table data.

**Next steps:**

1. Improve and test regex patterns against the actual OCR text.
2. Handle missing fields and multiple possible label formats.
3. Verify extracted values instead of assuming every regex match is correct.
4. Document the limitations of rule-based extraction.
5. Extend the extraction rules to other document types after invoice extraction is tested.


## Requirements

```bash
pip install tensorflow pandas scikit-learn matplotlib
```

---

## Reproducing

```bash
# 1. Generate manifest
python generate_manifest.py --dataset-root dataset --out manifest.csv

# 2. Train (Phase 2.6 combined)
python train_transfer_phase25.py

# 3. Evaluate
python evaluate.py --model transfer_model_phase25.keras --tag transfer_phase25
```
