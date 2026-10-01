# Document AI System — CNN Classification + OCR Extraction + Human Review

End-to-end document processing pipeline: classifies scanned documents into 5
types, extracts structured fields via OCR, and routes low-confidence results
to human review. Built as a placement portfolio project.

**Timeline:** Sept 20 – Oct 10, 2026 (21 days)

---

## Project phases

- [x] **Phase 1 — CNN baseline + dataset manifest** *(completed)*
- [x] **Phase 2 — Transfer learning (MobileNetV2)** *(completed)*
- [x] **Phase 2.5 — Modern data collection** *(completed)*
- [x] **Phase 2.6 — Combined retraining (old + modern)** *(completed)*
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