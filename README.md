# Document AI System — CNN Classification + OCR Extraction + Human Review

End-to-end document processing pipeline: classifies scanned documents into 5
types, extracts structured fields via OCR, and routes low-confidence results
to human review. Built as a placement portfolio project.

**Timeline:** Sept 20 – Oct 10, 2026 (21 days)

---

## Project phases

- [x] **Phase 1 — CNN baseline + dataset manifest** *(completed)*
- [x] **Phase 2 — Transfer learning (MobileNetV2)** *(completed)*
- [] **Phase 2.5 — Modern data augmentation + combined retraining** *(still going)*
- [ ] Phase 3 — OCR
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
| Invoice | 800 | ~300 | ~1,100 |
| Form | 800 | ~300 | ~1,100 |
| Resume | 800 | ~300 | ~1,100 |
| ID Card | 1,400 (synthetic) | 0 | 1,400 |
| Certificate | 600 (synthetic) | 0 | 600 |

**Modern sources:**
- **Resume:** Resumes-Images-Datasets (Kaggle, ~12.7k images)
- **Invoice:** High-Quality Invoice Images for OCR (Kaggle/Hugging Face, ~8.2k images)
- **Form:** CommonForms (Hugging Face) + synthetic HTML-rendered templates

**Rationale:** original training data was entirely historical microfilm scans
(RVL-CDIP) plus synthetic ID Card / Certificate images. Models trained only
on historical scans fail on modern clean documents due to domain shift.
Adding ~300 modern images per class teaches the model to recognize document
structure independent of style (scan noise vs clean render).

### Folder structure

```
dataset/
├── certificate/
├── form/
│   ├── rvl_cdip/        # original
│   └── modern/          # Phase 2.5 additions
├── id_card/
├── invoice/
│   ├── rvl_cdip/        # original
│   └── modern/          # Phase 2.5 additions
└── resume/
    ├── rvl_cdip/        # original
    └── modern/          # Phase 2.5 additions
```

---

## Phase 1 status

### Task 1 — Dataset manifest ✅ Complete

`generate_manifest.py` walks the `dataset/` folder and produces `manifest.csv`,
the single source of truth for train/val/test split membership used by every
later phase.

- Infers `source` (real/synthetic) from folder name
- Hashes every file to flag exact-duplicate images (`duplicates.csv`)
- 70/15/15 train/val/test split, stratified jointly by `class_name` + `source`

**Run:**
```bash
python generate_manifest.py --dataset-root dataset --out manifest.csv
```

**Output columns:** `filepath, filename, class_name, source, file_hash, split`

**Actual split achieved (Phase 1, original data only):**

| Class | Train | Val | Test |
|---|---|---|---|
| certificate | 420 | 90 | 90 |
| form | 560 | 120 | 120 |
| id_card | 980 | 210 | 210 |
| invoice | 560 | 120 | 120 |
| resume | 560 | 120 | 120 |

No exact-duplicate files detected.

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
human annotators and the model. The model confuses Invoice ↔ Form because
they share layout structure with no readable distinguishing text.

**Resolution improvement (160→320) yielded +5.3%; deepening fine-tuning
(30→50 layers) yielded +1.5%. Diminishing returns indicate the remaining
error is a data limitation, not a model capacity limitation.**

---

## Phase 2.5 status — Modern data augmentation

To improve performance on modern documents and break the RVL-CDIP ceiling,
~300 modern images per class were added for Invoice, Form, and Resume.

### Approach

- Combined old (RVL-CDIP) + modern images in a single `manifest.csv`
- Maintained class balance (~1,100 per class total for the 3 affected classes)
- Mixed old + modern rows in every split (train/val/test) to prevent domain leakage
- Retrained MobileNetV2 with identical hyperparameters to Phase 2

### Results (Phase 2.5, combined data)

*(to be filled after retraining)*

| Test set | Accuracy |
|---|---|
| RVL-CDIP only | TBD |
| Modern only | TBD |
| Mixed | TBD |

**Expected:** ~90–93% on modern documents; slight dip (1–2%) on RVL-CDIP-only.

---

## Requirements

```bash
pip install tensorflow pandas scikit-learn matplotlib
```

---

## Reproducing

```bash
# 1. Generate manifest
python generate_manifest.py --dataset-root dataset --out manifest.csv

# 2. Train
python train_transfer.py

# 3. Evaluate
python evaluate.py --model transfer_model.keras --tag transfer
```
