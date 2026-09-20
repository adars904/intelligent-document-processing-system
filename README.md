# Document AI System — CNN Classification + OCR Extraction + Human Review

End-to-end document processing pipeline: classifies scanned documents into 5
types, extracts structured fields via OCR, and routes low-confidence results
to human review. Built as a placement portfolio project.

**Timeline:** Sept 20 – Oct 10, 2026 (21 days)

## Project phases

- [x] **Phase 1 — CNN baseline + dataset manifest** *(in progress)*
- [ ] Phase 2 — Transfer learning
- [ ] Phase 3 — OCR
- [ ] Phase 4 — Extraction
- [ ] Phase 5 — Validation / confidence scoring
- [ ] Phase 6 — FastAPI + SQLite + model versioning + audit log
- [ ] Phase 7 — Streamlit dashboard
- [ ] Phase 8 — Testing + golden set + adversarial fixtures + Known Failure Modes doc
- [ ] Phase 9 — Integration + docs + Future Work (async job queue)

## Dataset

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

### Folder structure

```
dataset/
├── certificate/
├── form/
├── id_card/
├── invoice/
└── resume/
```

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

**Actual split achieved:**

| Class | Train | Val | Test |
|---|---|---|---|
| certificate | 420 | 90 | 90 |
| form | 560 | 120 | 120 |
| id_card | 980 | 210 | 210 |
| invoice | 560 | 120 | 120 |
| resume | 560 | 120 | 120 |

No exact-duplicate files detected.

### Task 2 — CNN baseline (from scratch) ⏳ Next

Plain CNN, no pretrained weights, to establish a floor accuracy before
Phase 2's transfer-learning model. Reads `manifest.csv` directly so
train/val/test membership stays fixed across every phase.

## Requirements

```bash
pip install tensorflow pandas scikit-learn matplotlib
```