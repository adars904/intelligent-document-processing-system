"""resume_extractor.py - single-file Resume OCR + information extraction.

Pipeline: load (PDF/JPG/PNG) -> OpenCV preprocessing -> Tesseract OCR (column-aware) ->
section detection -> extraction -> normalisation -> validation -> structured JSON.

Requires: numpy, opencv-python, Pillow, pytesseract, pdf2image (+ Tesseract and Poppler installed).

Usage:
    from resume_extractor import process_resume
    result = process_resume("resume.pdf")          # path or bytes
    python resume_extractor.py resume.pdf          # prints JSON
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import os
import shutil
from dataclasses import dataclass, field
import io
from pathlib import Path
from typing import List, Optional, Union
import cv2
import numpy as np
import re
from datetime import date
from typing import List
import statistics
import unicodedata
from typing import List, Optional
import difflib
from typing import Dict, List, Optional
from typing import Dict, List
from typing import Any, Dict, List
import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path  # noqa: E402


# ==============================================================================
# MODELS
# ==============================================================================
@dataclass
class Line:
    """One reconstructed text line.

    `height` (pixels) and `conf` (0-100) come from OCR and are optional so the
    extractors can also be unit-tested on plain text.
    """

    text: str
    page: int = 0
    height: Optional[float] = None
    conf: Optional[float] = None

    @property
    def is_blank(self) -> bool:
        return not self.text.strip()


def lines_from_text(text: str) -> list[Line]:
    """Build Line objects from plain text (used for tests / already-digital text)."""
    return [Line(t) for t in text.splitlines()]


# ==============================================================================
# CONFIG
# ==============================================================================
_WIN_TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def _default_tesseract() -> Optional[str]:
    env = os.environ.get("TESSERACT_CMD")
    if env:
        return env
    if shutil.which("tesseract"):
        return None  # already on PATH, pytesseract will find it
    if os.path.exists(_WIN_TESSERACT):
        return _WIN_TESSERACT
    return None


@dataclass
class Settings:
    # --- loading ---
    pdf_dpi: int = 300                 # Tesseract works best around 300 DPI
    max_pages: int = 8                 # resumes rarely exceed this; protects memory
    max_file_mb: float = 20.0
    poppler_path: Optional[str] = field(default_factory=lambda: os.environ.get("POPPLER_PATH"))

    # --- preprocessing ---
    min_width: int = 1600              # upscale smaller images up to this width
    max_width: int = 3600              # downscale huge phone photos
    deskew: bool = True
    fix_dark_regions: bool = True      # invert dark sidebars so white text becomes black
    binarize: str = "otsu"             # "otsu" | "adaptive" | "none"

    # --- OCR ---
    tesseract_cmd: Optional[str] = field(default_factory=_default_tesseract)
    lang: str = "eng"
    psm_primary: int = 3               # auto segmentation; PSM 6 dropped large header text (see README)
    psm_fallback: int = 6
    retry_below_conf: float = 60.0     # try the other PSM if mean conf is below this
    min_word_conf: float = 10.0        # drop obvious garbage words
    detect_columns: bool = True
    try_rotations_below_conf: float = 35.0   # last-resort 90/180/270 attempt

    # --- validation ---
    min_text_chars: int = 40
    good_ocr_conf: float = 90.0   # clean renders scored >=92, damaged ones 84-88 in our samples


# ==============================================================================
# LOADER
# ==============================================================================
SUPPORTED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png"}


class ResumeLoadError(Exception):
    """Raised for unsupported, empty, oversized or corrupted input files."""


@dataclass
class LoadedResume:
    images: List[np.ndarray]          # BGR uint8 images, one per page
    file_type: str                    # "pdf" | "jpg" | "png"
    source_name: str
    total_pages: int                  # pages in the original document
    truncated: bool = False           # True if max_pages cut the document short


def _sniff_type(data: bytes, filename: Optional[str]) -> str:
    """Trust magic bytes over the extension (uploads are often mislabeled)."""
    if data[:5] == b"%PDF-":
        return "pdf"
    if data[:3] == b"\xff\xd8\xff":
        return "jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    ext = Path(filename).suffix.lower() if filename else ""
    if ext in SUPPORTED_EXTENSIONS:
        raise ResumeLoadError(
            f"File content does not look like a valid {ext} file (corrupted or mislabeled)."
        )
    raise ResumeLoadError(
        f"Unsupported file type '{ext or 'unknown'}'. Supported: PDF, JPG, JPEG, PNG."
    )


def _render_with_pymupdf(pdf_bytes: bytes, settings: Settings) -> tuple[List[np.ndarray], int]:
    """Render PDF pages with PyMuPDF. Raises ImportError if PyMuPDF is not installed."""
    try:
        import pymupdf as fitz  # PyMuPDF (new name)
    except ImportError:
        import fitz  # older PyMuPDF versions

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ResumeLoadError(f"PDF is corrupted or unreadable: {exc}") from exc
    if doc.needs_pass:
        raise ResumeLoadError("PDF is password-protected/encrypted.")
    total = doc.page_count
    if total < 1:
        raise ResumeLoadError("PDF has no pages.")
    zoom = settings.pdf_dpi / 72.0
    images = []
    for i in range(min(total, settings.max_pages)):
        pix = doc[i].get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csRGB, alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        images.append(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    return images, total


def convert_pdf_to_images(pdf_bytes: bytes, settings: Settings) -> tuple[List[np.ndarray], int]:
    """Render PDF pages to BGR images with pdf2image (needs Poppler).

    Returns (images, total_page_count). Only the first `max_pages` are rendered.
    """
    try:
        from pdf2image import convert_from_bytes, pdfinfo_from_bytes
        from pdf2image.exceptions import (
            PDFInfoNotInstalledError,
            PDFPageCountError,
            PDFSyntaxError,
        )
    except ImportError as exc:  # pragma: no cover
        raise ResumeLoadError("pdf2image is not installed.") from exc

    kwargs = {"poppler_path": settings.poppler_path} if settings.poppler_path else {}
    try:
        info = pdfinfo_from_bytes(pdf_bytes, **kwargs)
        total = int(info.get("Pages", 0))
        if info.get("Encrypted", "no").lower().startswith("yes"):
            raise ResumeLoadError("PDF is password-protected/encrypted.")
        if total < 1:
            raise ResumeLoadError("PDF has no pages.")
        pil_pages = convert_from_bytes(
            pdf_bytes, dpi=settings.pdf_dpi, first_page=1,
            last_page=min(total, settings.max_pages), **kwargs,
        )
    except PDFInfoNotInstalledError as exc:
        # Poppler missing: fall back to PyMuPDF (pure pip install, no external program)
        try:
            return _render_with_pymupdf(pdf_bytes, settings)
        except ImportError:
            raise ResumeLoadError(
                "Poppler is not installed or not on PATH, and PyMuPDF is not installed. "
                "Easiest fix: pip install pymupdf  (or install Poppler and set POPPLER_PATH)."
            ) from exc
    except (PDFPageCountError, PDFSyntaxError) as exc:
        raise ResumeLoadError(f"PDF is corrupted or unreadable: {exc}") from exc

    images = [cv2.cvtColor(np.array(p.convert("RGB")), cv2.COLOR_RGB2BGR) for p in pil_pages]
    return images, total


def _decode_image(data: bytes) -> np.ndarray:
    # imdecode works with bytes (and therefore with non-ASCII Windows paths,
    # which cv2.imread cannot open).
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ResumeLoadError("Image could not be decoded (corrupted file).")
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:  # PNG with alpha: composite on white, not black
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        rgb = img[:, :, :3].astype(np.float32)
        return (rgb * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    return img


def load_resume(
    source: Union[str, os.PathLike, bytes],
    filename: Optional[str] = None,
    settings: Optional[Settings] = None,
) -> LoadedResume:
    """Load a resume from a path or from raw bytes (e.g. a FastAPI upload)."""
    settings = settings or Settings()

    if isinstance(source, (bytes, bytearray)):
        data = bytes(source)
        name = filename or "upload"
    else:
        path = Path(source)
        if not path.is_file():
            raise ResumeLoadError(f"File not found: {path}")
        data = path.read_bytes()
        name = filename or path.name

    if not data:
        raise ResumeLoadError("File is empty.")
    if len(data) > settings.max_file_mb * 1024 * 1024:
        raise ResumeLoadError(f"File exceeds {settings.max_file_mb} MB limit.")

    ftype = _sniff_type(data, name)
    if ftype == "pdf":
        images, total = convert_pdf_to_images(data, settings)
        return LoadedResume(images, "pdf", name, total, truncated=total > len(images))
    return LoadedResume([_decode_image(data)], ftype, name, 1)


# ==============================================================================
# PREPROCESSING
# ==============================================================================
def _resize_to_range(img: np.ndarray, min_w: int, max_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w < min_w:
        scale = min_w / w
        return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if w > max_w:
        scale = max_w / w
        return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


def invert_dark_regions(gray: np.ndarray, block: int = 16, dark_thresh: int = 130,
                        min_area_frac: float = 0.02) -> np.ndarray:
    """Invert large dark areas (e.g. a navy sidebar with white text).

    Tesseract reads dark-on-light far better than light-on-dark. We average
    intensity over coarse blocks (so thin text strokes don't matter), find big
    connected dark regions, and invert only those pixels. Ordinary black text on
    white has block means near 200+, so it is never touched.
    """
    h, w = gray.shape
    small = cv2.resize(gray, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_AREA)
    mask_small = (small < dark_thresh).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_small, connectivity=4)
    keep = np.zeros_like(mask_small)
    min_area = min_area_frac * mask_small.size
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 1
    if not keep.any():
        return gray
    mask = cv2.resize(keep, (w, h), interpolation=cv2.INTER_NEAREST)
    mask = cv2.dilate(mask, np.ones((block, block), np.uint8))
    out = gray.copy()
    out[mask > 0] = 255 - out[mask > 0]
    return out


def estimate_skew_angle(gray: np.ndarray, max_angle: float = 10.0) -> float:
    """Projection-profile skew estimate (degrees). Robust across OpenCV versions."""
    h, w = gray.shape
    scale = 800.0 / max(w, 1)
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    _, bw = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if cv2.countNonZero(bw) < 200:
        return 0.0
    sh, sw = bw.shape
    center = (sw / 2, sh / 2)

    def score(angle: float) -> float:
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        rot = cv2.warpAffine(bw, m, (sw, sh), flags=cv2.INTER_NEAREST)
        proj = rot.sum(axis=1, dtype=np.float64)
        return float(np.sum(np.diff(proj) ** 2))

    coarse = np.arange(-max_angle, max_angle + 0.01, 1.0)
    best = max(coarse, key=score)
    fine = np.arange(best - 0.9, best + 0.91, 0.1)
    best = max(fine, key=score)
    # only rotate if it is a clear improvement over "no rotation"
    return float(best) if score(best) > 1.05 * score(0.0) else 0.0


def deskew(gray: np.ndarray) -> np.ndarray:
    angle = estimate_skew_angle(gray)
    if abs(angle) < 0.3:
        return gray
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def preprocess_image(image: np.ndarray, settings: Settings | None = None) -> np.ndarray:
    """BGR/gray image -> cleaned single-channel image ready for Tesseract."""
    settings = settings or Settings()
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = _resize_to_range(gray, settings.min_width, settings.max_width)

    if settings.fix_dark_regions:
        gray = invert_dark_regions(gray)
    if settings.deskew:
        gray = deskew(gray)

    # light denoise that keeps thin strokes (fastNlMeans is slow and blurs small text)
    gray = cv2.medianBlur(gray, 3)

    if settings.binarize == "otsu":
        _, out = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return out
    if settings.binarize == "adaptive":  # uneven lighting (phone photos)
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 41, 15)
    return gray


# ==============================================================================
# NORMALIZERS
# ==============================================================================
# =============================================================================== dates
MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_MONTH = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|"
          r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)")
_YEAR = r"(?:19|20)\d{2}"
PRESENT = (r"(?:Present|Current(?:ly)?|Till\s+(?:date|now|present)|To\s+date|Now|"
           r"Ongoing|Today|Continuing)")
DATE_TOKEN = (
    rf"(?:{_MONTH}\.?,?\s*(?:{_YEAR}|'\d{{2}})"   # Jan 2020 / Jan '20
    rf"|\d{{1,2}}\s*[/.\-]\s*{_YEAR}"             # 01/2020
    rf"|{_YEAR}\s*[/\-]\s*\d{{1,2}}(?!\d)"        # 2020-01
    rf"|{_YEAR})"                                  # 2020
)
_SEP = r"\s*(?:[-\u2010-\u2015]|\bto\b|\buntil\b|\btill\b|\bthrough\b)\s*"

RANGE_RE = re.compile(
    rf"(?<![\d/])(?P<start>{DATE_TOKEN}){_SEP}(?P<end>{DATE_TOKEN}|{PRESENT})(?!\d)",
    re.IGNORECASE,
)
# "Jun - Aug 2021": the start has no year, borrow it from the end
MONTH_RANGE_RE = re.compile(
    rf"(?P<sm>{_MONTH})\.?{_SEP}(?P<end>{_MONTH}\.?,?\s*(?P<ey>{_YEAR}))(?!\d)",
    re.IGNORECASE,
)
SINGLE_DATE_RE = re.compile(rf"(?<![\d/]){DATE_TOKEN}(?!\d)", re.IGNORECASE)
YEAR_RE = re.compile(rf"(?<!\d)({_YEAR})(?!\d)")


def parse_date_token(tok: str) -> Optional[str]:
    """'Jan 2020' -> '2020-01', '2020' -> '2020', 'Present' -> 'Present'."""
    if not tok:
        return None
    t = tok.strip().strip(",.")
    if re.fullmatch(PRESENT, t, re.IGNORECASE):
        return "Present"
    m = re.fullmatch(rf"({_MONTH})\.?,?\s*({_YEAR}|'\d{{2}})", t, re.IGNORECASE)
    if m:
        month = MONTHS[m.group(1)[:3].lower()]
        y = m.group(2)
        year = int(y) if len(y) == 4 else 2000 + int(y[1:])
        return f"{year:04d}-{month:02d}"
    m = re.fullmatch(rf"(\d{{1,2}})\s*[/.\-]\s*({_YEAR})", t)
    if m and 1 <= int(m.group(1)) <= 12:
        return f"{int(m.group(2)):04d}-{int(m.group(1)):02d}"
    m = re.fullmatch(rf"({_YEAR})\s*[/\-]\s*(\d{{1,2}})", t)
    if m and 1 <= int(m.group(2)) <= 12:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}"
    if re.fullmatch(_YEAR, t):
        return t
    return None


_PRESENT_TYPO = re.compile(rf"((?:{DATE_TOKEN})\s*(?:[-\u2010-\u2015]|\bto\b)\s*)([A-Za-z]{{6,8}})\b", re.I)


def _fix_present(text: str) -> str:
    """OCR often garbles 'Present' (Preeent, Prosent). Repair it only right after a date + dash."""
    import difflib

    def repl(m):
        word = m.group(2)
        if word.lower() in MONTHS or word[:3].lower() in MONTHS:
            return m.group(0)
        close = difflib.get_close_matches(word.lower(), ["present", "current", "ongoing"], n=1, cutoff=0.74)
        return m.group(1) + (close[0].capitalize() if close else word)
    return _PRESENT_TYPO.sub(repl, text)


def find_date_range(text: str):
    """Return (start, end, (span_start, span_end)) normalised, or None."""
    text = _fix_present(text)
    m = RANGE_RE.search(text)
    if m:
        s, e = parse_date_token(m.group("start")), parse_date_token(m.group("end"))
        if s and e:
            return s, e, m.span()
    m = MONTH_RANGE_RE.search(text)
    if m:
        year = m.group("ey")
        s = parse_date_token(f"{m.group('sm')} {year}")
        e = parse_date_token(m.group("end"))
        if s and e:
            return s, e, m.span()
    return None


def strip_date_range(text: str) -> str:
    r = find_date_range(text)
    if not r:
        return text
    a, b = r[2]
    return (text[:a] + " " + text[b:]).strip()


def date_sort_key(d: Optional[str]) -> Optional[tuple]:
    """Comparable key; 'Present' sorts as today. None if unparsable."""
    if not d:
        return None
    if d == "Present":
        t = date.today()
        return (t.year, t.month)
    m = re.fullmatch(r"(\d{4})(?:-(\d{2}))?", d)
    return (int(m.group(1)), int(m.group(2) or 1)) if m else None


# =============================================================================== email
EMAIL_RE = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9_%+\-]*(?:\s?\.\s?[A-Za-z0-9_%+\-]+)*\s?@\s?[A-Za-z0-9][A-Za-z0-9\-]*(?:\s?\.\s?[A-Za-z0-9\-]+)*"
    r"\s?\.\s?[A-Za-z]{2,}"
)
_EMAIL_STRICT = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}$")


def normalize_email(raw: str) -> Optional[str]:
    """Remove OCR-inserted spaces, lowercase, and verify the shape."""
    e = re.sub(r"\s+", "", raw).strip(".,;:()<>[]").lower()
    return e if _EMAIL_STRICT.match(e) else None


def find_emails(text: str) -> list[str]:
    out = []
    for m in EMAIL_RE.finditer(text):
        e = normalize_email(m.group(0))
        if e and e not in out:
            out.append(e)
    return out


# =============================================================================== phone
PHONE_RE = re.compile(r"(?<![\w.])(?:\+\s?\d{1,3}[\s\-.]?)?(?:\(\d{2,5}\)[\s\-.]?)?\d[\d\s\-.]{6,16}\d(?![\w])")
_YEARS_ONLY = re.compile(rf"(?:{_YEAR}[\s\-\u2013.]*)+")


def normalize_phone(raw: str) -> Optional[str]:
    """Return '+<digits>' when a country code is present, else the bare digits.

    We never invent a country code. Valid length: 10-15 digits (E.164 max 15).
    """
    s = raw.strip()
    if _YEARS_ONLY.fullmatch(s):
        return None
    digits = re.sub(r"\D", "", s)
    if s.lstrip().startswith("+"):
        return "+" + digits if 10 <= len(digits) <= 15 else None
    if digits.startswith("00") and 12 <= len(digits) <= 17:
        return "+" + digits[2:]
    if len(digits) == 12 and digits.startswith("91"):      # "91 98765 43210"
        return "+" + digits
    if len(digits) == 11 and digits.startswith("0"):       # trunk prefix
        digits = digits[1:]
    return digits if 10 <= len(digits) <= 13 else None


def find_phones(text: str) -> list[str]:
    out = []
    for m in PHONE_RE.finditer(text):
        p = normalize_phone(m.group(0))
        if p and p not in out:
            out.append(p)
    return out


# =============================================================================== urls
_LINKEDIN_RE = re.compile(
    r"(?:https?://)?(?:[a-z]{2,3}\.)?linkedin\.com/(?:in|pub)/[A-Za-z0-9\-_%.]+/?", re.I)
_GITHUB_RE = re.compile(
    r"(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9\-_]+)(?:/[A-Za-z0-9\-_.]+)?/?", re.I)
_GENERIC_URL_RE = re.compile(
    r"(?<![@\w.\-/])(?:(?:https?://|www\.)[^\s,;|()<>\"']+|"
    r"(?:[a-z0-9\-]+\.)+(?:com|io|dev|me|net|org|co|app|tech|xyz|site|online|ai|design|codes|page|in)"
    r"(?:/[^\s,;|()<>\"']*)?(?![\w@]))",
    re.I,
)
_NOT_PORTFOLIO = ("linkedin.com", "github.com", "gmail.com", "yahoo.com", "outlook.com",
                  "hotmail.com", "mailto:", "asp.net", "vb.net", "node.js", "gitlab.com",
                  "leetcode.com", "hackerrank.com", "kaggle.com", "twitter.com", "x.com",
                  "medium.com", "stackoverflow.com")


def normalize_url(raw: str) -> str:
    u = raw.strip().strip(".,;:()<>[]\"'")
    u = re.sub(r"\s+", "", u)
    if not re.match(r"^https?://", u, re.I):
        u = "https://" + re.sub(r"^www\.", "www.", u, flags=re.I)
    return u.rstrip("/")


def _fix_url_spaces(text: str) -> str:
    """OCR often inserts spaces around '/' and '.' in profile links."""
    text = re.sub(r"(?i)(linkedin\s*\.\s*com|github\s*\.\s*com)\s*/\s*(in\s*/\s*|pub\s*/\s*)?",
                  lambda m: re.sub(r"\s+", "", m.group(1)) + "/" + (re.sub(r"\s+", "", m.group(2)) if m.group(2) else ""),
                  text)
    return text


def find_linkedin(text: str) -> Optional[str]:
    text = _fix_url_spaces(text)
    m = _LINKEDIN_RE.search(text)
    return normalize_url(m.group(0)) if m else None


def find_github_profile(text: str) -> Optional[str]:
    """Return the profile URL (github.com/<user>) - never a repository link."""
    text = _fix_url_spaces(text)
    for m in _GITHUB_RE.finditer(text):
        user = m.group(1)
        if user.lower() in {"features", "topics", "about", "login", "pricing"}:
            continue
        return f"https://github.com/{user}"
    return None


def find_portfolio(text: str, exclude_emails: bool = True) -> Optional[str]:
    if exclude_emails:
        text = EMAIL_RE.sub(" ", text)
    for m in _GENERIC_URL_RE.finditer(text):
        cand = m.group(0)
        if any(bad in cand.lower() for bad in _NOT_PORTFOLIO):
            continue
        return normalize_url(cand)
    return None


# =============================================================================== GPA / percent
_NUM = r"(\d{1,3}(?:\.\d{1,3})?)"
_LABEL = (r"(CGPA|C\.G\.P\.A\.?|GPA|CPI|SGPA|DGPA|Grade\s+Point(?:\s+Average)?|"
          r"Aggregate|Percentage|Percent|Marks|Score|Result)")
_LABEL_FIRST = re.compile(
    rf"\b{_LABEL}\s*(?:of|is|:|-|=)?\s*{_NUM}\s*(?:/\s*(\d{{1,3}}(?:\.\d+)?)|(%|percent))?", re.I)
_NUM_FIRST = re.compile(rf"(?<![\d.]){_NUM}\s*(?:/\s*(\d{{1,3}}(?:\.\d+)?)\s*)?(CGPA|GPA|CPI)\b", re.I)
_PCT = re.compile(r"(?<![\d.])(\d{2,3}(?:\.\d{1,2})?)\s*(?:%|percent\b)", re.I)


def _clean_num(s: str) -> str:
    return str(float(s)).rstrip("0").rstrip(".") if "." in s else s


def find_grade(text: str) -> Optional[str]:
    """Return e.g. '8.5/10', '3.8/4.0', '8.5' (scale unknown) or '85%'. None if absent."""
    m = _LABEL_FIRST.search(text)
    if m:
        label, val, scale, pct = m.group(1).lower(), float(m.group(2)), m.group(3), m.group(4)
        is_pct_label = label.startswith(("perc", "aggreg", "marks", "score", "result"))
        if pct or (is_pct_label and not scale and val > 10):
            return f"{_clean_num(m.group(2))}%" if val <= 100 else None
        if scale:
            sc = float(scale)
            return f"{_clean_num(m.group(2))}/{_clean_num(scale)}" if 0 < val <= sc else None
        return _clean_num(m.group(2)) if 0 < val <= 10 else None
    m = _NUM_FIRST.search(text)
    if m:
        val = float(m.group(1))
        if m.group(2):
            sc = float(m.group(2))
            return f"{_clean_num(m.group(1))}/{_clean_num(m.group(2))}" if 0 < val <= sc else None
        return _clean_num(m.group(1)) if 0 < val <= 10 else None
    m = _PCT.search(text)
    if m and float(m.group(1)) <= 100:
        return f"{_clean_num(m.group(1))}%"
    return None


# ==============================================================================
# VOCAB
# ==============================================================================
PL, FW, TOOLS, TECH = ("programming_languages", "frameworks_and_libraries",
                       "tools_and_technologies", "technical_skills")

_RAW = {
    PL: [
        "Python", "Java", "JavaScript", "TypeScript", "C", "C++", "C#", "Go", "Golang", "Rust",
        "Kotlin", "Swift", "Ruby", "PHP", "Scala", "R", "MATLAB", "Perl", "Dart", "Objective-C",
        "SQL", "PL/SQL", "T-SQL", "HTML", "CSS", "HTML5", "CSS3", "Bash", "Shell", "PowerShell",
        "Assembly", "Julia", "Haskell", "Lua", "VBA", "Groovy", "SAS", "Solidity", "Verilog",
    ],
    FW: [
        "React", "React Native", "Angular", "Vue.js", "Next.js", "Node.js", "Express.js", "Django",
        "Flask", "FastAPI", "Spring", "Spring Boot", "Hibernate", ".NET", "ASP.NET", "Laravel",
        "Rails", "jQuery", "Bootstrap", "Tailwind CSS", "Redux", "Svelte", "Flutter",
        "TensorFlow", "PyTorch", "Keras", "scikit-learn", "Pandas", "NumPy", "SciPy", "Matplotlib",
        "Seaborn", "Plotly", "OpenCV", "NLTK", "spaCy", "Hugging Face", "Transformers", "LangChain",
        "XGBoost", "LightGBM", "Streamlit", "Selenium", "PyTest", "JUnit", "Celery", "SQLAlchemy",
        "Pydantic", "Pillow", "BeautifulSoup", "Scrapy", "Jest", "Mocha", "GraphQL", "gRPC",
        "Apache Spark", "PySpark", "Hadoop", "Airflow", "Kafka", "Pytesseract", "Tkinter", "Qt",
    ],
    TOOLS: [
        "Git", "GitHub", "GitLab", "Bitbucket", "Docker", "Kubernetes", "Jenkins", "Terraform",
        "Ansible", "AWS", "Azure", "GCP", "Google Cloud", "Heroku", "Linux", "Unix", "Windows",
        "MySQL", "PostgreSQL", "MongoDB", "SQLite", "Oracle", "Redis", "Elasticsearch", "Cassandra",
        "DynamoDB", "Firebase", "Snowflake", "BigQuery", "Tableau", "Power BI", "Excel",
        "MS Excel", "MS Office", "Jira", "Confluence", "Postman", "Swagger", "VS Code", "PyCharm",
        "IntelliJ", "Eclipse", "Jupyter", "Google Colab", "Figma", "Photoshop", "Nginx", "Apache",
        "RabbitMQ", "Grafana", "Prometheus", "Tesseract", "Tesseract OCR", "Unity", "Android Studio",
        "Maven", "Gradle", "npm", "Webpack", "Vercel", "Netlify", "Splunk", "ServiceNow", "SAP",
        "Salesforce", "Hive", "Databricks", "MLflow", "DVC", "Slack", "Trello", "Arduino",
        "Raspberry Pi", "AutoCAD", "SolidWorks", "Vim", "MERN Stack", "MEAN Stack", "LAMP Stack",
    ],
    TECH: [
        "Machine Learning", "Deep Learning", "Natural Language Processing", "Computer Vision",
        "Data Structures", "Algorithms", "Data Structures and Algorithms", "Object-Oriented Programming",
        "OOP", "REST APIs", "RESTful APIs", "Microservices", "System Design", "Data Analysis",
        "Data Analytics", "Data Visualization", "Data Science", "Statistics", "Statistical Modeling",
        "Agile", "Scrum", "CI/CD", "DevOps", "Cloud Computing", "Web Development", "OCR",
        "Image Processing", "Generative AI", "LLMs", "Reinforcement Learning", "Big Data",
        "ETL", "Data Mining", "Data Engineering", "Database Management", "Operating Systems",
        "Computer Networks", "Software Testing", "Unit Testing", "Test Automation", "API Development",
        "Cybersecurity", "Neural Networks", "Time Series Analysis", "A/B Testing", "Feature Engineering",
        "Mobile Development", "Embedded Systems", "IoT", "Blockchain", "UI/UX", "Version Control",
        "Responsive Design", "Object Detection", "Predictive Modeling", "Data Wrangling",
    ],
}

_ALIASES = {
    "js": "JavaScript", "ts": "TypeScript", "node": "Node.js", "nodejs": "Node.js",
    "node js": "Node.js", "react.js": "React", "reactjs": "React", "react js": "React",
    "vue": "Vue.js", "vuejs": "Vue.js", "angularjs": "Angular", "express": "Express.js",
    "expressjs": "Express.js", "next": "Next.js", "nextjs": "Next.js", "sklearn": "scikit-learn",
    "scikit learn": "scikit-learn", "scikitlearn": "scikit-learn", "tensor flow": "TensorFlow",
    "pytorch": "PyTorch", "open cv": "OpenCV", "opencv-python": "OpenCV", "cv2": "OpenCV",
    "postgres": "PostgreSQL", "postgresql": "PostgreSQL", "mongo": "MongoDB", "mongo db": "MongoDB",
    "my sql": "MySQL", "k8s": "Kubernetes", "amazon web services": "AWS", "gcp": "GCP",
    "google cloud platform": "GCP", "microsoft azure": "Azure", "ms-excel": "MS Excel",
    "microsoft excel": "MS Excel", "microsoft office": "MS Office", "ml": "Machine Learning",
    "dl": "Deep Learning", "nlp": "Natural Language Processing", "cv": "Computer Vision",
    "dsa": "Data Structures and Algorithms", "oops": "Object-Oriented Programming",
    "object oriented programming": "Object-Oriented Programming", "rest api": "REST APIs",
    "rest apis": "REST APIs", "restful api": "REST APIs", "restful apis": "REST APIs",
    "rest": "REST APIs", "ci cd": "CI/CD", "cicd": "CI/CD", "powerbi": "Power BI",
    "power-bi": "Power BI", "py": "Python", "golang": "Go", "c plus plus": "C++", "cpp": "C++",
    "vscode": "VS Code", "visual studio code": "VS Code", "huggingface": "Hugging Face",
    "bs4": "BeautifulSoup", "beautiful soup": "BeautifulSoup", "tailwind": "Tailwind CSS",
    "tailwindcss": "Tailwind CSS", "spring-boot": "Spring Boot", "springboot": "Spring Boot",
    "dot net": "ASP.NET", "pyspark": "PySpark", "numpy": "NumPy", "pandas": "Pandas",
    "ms office": "MS Office", "google colab": "Google Colab", "colab": "Google Colab",
    "tesseract-ocr": "Tesseract OCR", "pytesseract": "Pytesseract", "restful": "REST APIs",
    "html/css": None,  # handled by slash splitting
}

CATEGORY_OF: dict[str, str] = {}
CANONICAL: dict[str, str] = {}


def _key(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


for _cat, _names in _RAW.items():
    for _n in _names:
        CATEGORY_OF[_n] = _cat
        CANONICAL[_key(_n)] = _n
for _a, _target in _ALIASES.items():
    if _target:
        CANONICAL[_key(_a)] = _target

SOFT_SKILLS = {
    "communication", "teamwork", "leadership", "problem solving", "problem-solving",
    "time management", "critical thinking", "adaptability", "creativity", "collaboration",
    "interpersonal skills", "analytical skills", "analytical thinking", "decision making",
    "work ethic", "multitasking", "presentation skills", "public speaking", "negotiation",
    "team player", "quick learner", "hard working", "self motivated", "management",
    "conflict resolution", "emotional intelligence", "attention to detail", "punctuality",
}

SPOKEN_LANGUAGES = {
    "english", "hindi", "kannada", "tamil", "telugu", "malayalam", "marathi", "bengali", "gujarati",
    "punjabi", "urdu", "odia", "spanish", "french", "german", "chinese", "mandarin", "japanese",
    "korean", "arabic", "russian", "portuguese", "italian", "dutch",
}


def canonicalize(token: str) -> str | None:
    """Return the canonical skill name if `token` is a known skill, else None."""
    k = _key(token)
    k = re.sub(r"^(?:basic|advanced|intermediate|proficient in|knowledge of)\s+", "", k)
    return CANONICAL.get(k)


def category_of(canonical: str) -> str:
    return CATEGORY_OF.get(canonical, TECH)


# Longest-first alternation of every known term for scanning short, trusted blocks
# (e.g. one project's text). Word-boundary guards stop "R" or "C" matching inside words.
_terms = sorted({*CATEGORY_OF.keys(), *[a for a in CANONICAL if a not in {_key(c) for c in CATEGORY_OF}]},
                key=len, reverse=True)
_AMBIGUOUS = {"r", "c", "go", "rest", "spring", "unity", "oracle", "apache", "swift", "ruby",
              "next", "node", "cv", "dl", "ml", "py", "js", "ts", "vue", "express", "windows",
              "unix", "shell", "assembly", "excel", "agile", "scrum", "statistics", "algorithms",
              "management", "react", "selenium", "transformers", "pillow", "sas", "qt", "dart", "sap"}
_scan_terms = [t for t in _terms if _key(t) not in _AMBIGUOUS or t[:1].isupper()]
SCAN_RE = re.compile(
    r"(?<![A-Za-z0-9+#.])(" + "|".join(re.escape(t) for t in _scan_terms) + r")(?![A-Za-z0-9+#])",
    re.IGNORECASE,
)


def scan_known_skills(text: str) -> list[str]:
    """Find known skills in a trusted block of text. Ambiguous words (R, C, Go, Swift...)
    are matched only when written with their canonical capitalisation."""
    found: list[str] = []
    for m in SCAN_RE.finditer(text):
        raw = m.group(1)
        canon = CANONICAL.get(_key(raw))
        if not canon:
            continue
        if _key(canon) in _AMBIGUOUS and raw != canon:
            continue
        if canon not in found:
            found.append(canon)
    return found


def canonicalize_fuzzy(token: str) -> str | None:
    """Exact match first, then a conservative fuzzy match (OCR near-misses like 'FastAP!')."""
    exact = canonicalize(token)
    if exact:
        return exact
    import difflib
    k = re.sub(r"[^a-z0-9+#.\- ]", "i", _key(token))          # '!' / '|' are usually a misread 'i'/'l'
    if len(k) < 6:
        return None
    close = difflib.get_close_matches(k, list(CANONICAL), n=1, cutoff=0.86)
    return CANONICAL[close[0]] if close else None


# ==============================================================================
# TEXT_UTILS
# ==============================================================================
# Real bullets plus the glyphs OCR commonly produces for them.
_BULLET_RE = re.compile(r"^\s*(?:[•●▪■◦○·*►▶➢✓✔▸»«©®°¢+]|[-–—](?=\s)|[eo](?=\s+\S)|\d{1,2}[.)](?=\s))\s*")
_HARD_BULLET_RE = re.compile(r"^\s*(?:[•●▪■◦○·*►▶➢✓✔▸»«©®¢]|[-–—](?=\s)|\+(?=\s))")
_STOP = {"of", "and", "the", "at", "in", "for", "to", "a", "an", "&", "on", "with", "-", "|"}


def is_bullet(line: str) -> bool:
    return bool(_BULLET_RE.match(line)) and bool(_BULLET_RE.sub("", line).strip())


def is_hard_bullet(line: str) -> bool:
    """Unambiguous bullet glyph (excludes the 'e '/'o ' OCR guesses and numbering)."""
    return bool(_HARD_BULLET_RE.match(line))


def strip_bullet(line: str) -> str:
    return _BULLET_RE.sub("", line, count=1).strip()


def title_ratio(text: str) -> float:
    """Share of content words that start with an uppercase letter or are ALL CAPS."""
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z.&'\-]*", text) if w.lower() not in _STOP]
    if not words:
        return 0.0
    return sum(1 for w in words if w[0].isupper()) / len(words)


def split_segments(text: str, split_at: bool = True) -> List[str]:
    """Split a header line on visual separators (|, bullets, wide gaps, ' - ', '@', ' at X')."""
    pattern = r"\s*[|•·●▪]\s*|\s{3,}|\s+[-–—]\s+|\s+@\s+"
    if split_at:
        pattern += r"|\s+at\s+(?=[A-Z])"
    parts = re.split(pattern, text)
    return [p.strip(" ,;:-–—") for p in parts if p and p.strip(" ,;:-–—")]


def merge_wrapped(lines: List[str]) -> List[str]:
    """Join physically wrapped lines into logical items.

    A new item starts at a bullet glyph or a blank line. Otherwise, if the previous
    line is 'long' (near the column width) or the current one starts lowercase, the
    current line is a continuation. With no bullets at all, each line is its own item
    unless it clearly continues the previous one.
    """
    nonblank = [l for l in lines if l.strip()]
    if not nonblank:
        return []
    max_len = max(len(strip_bullet(l)) for l in nonblank)
    any_bullets = any(is_bullet(l) for l in nonblank)
    items: List[str] = []
    cur = ""
    prev_len = 0
    for l in lines:
        if not l.strip():
            if cur:
                items.append(cur)
                cur = ""
            continue
        starts_new = is_bullet(l) if any_bullets else True
        txt = strip_bullet(l)
        continues = bool(cur) and (
            txt[:1].islower()
            or (prev_len >= 0.8 * max_len and not cur.rstrip().endswith((".", ";")) and not starts_new)
        )
        if continues and not (any_bullets and is_bullet(l)):
            cur = f"{cur} {txt}".strip()
        else:
            if cur:
                items.append(cur)
            cur = txt
        prev_len = len(txt)
    if cur:
        items.append(cur)
    return items


# ==============================================================================
# OCR
# ==============================================================================
class OCRError(Exception):
    pass


@dataclass
class Word:
    text: str
    x0: int
    y0: int
    x1: int
    y1: int
    conf: float

    @property
    def h(self) -> int:
        return self.y1 - self.y0

    @property
    def yc(self) -> float:
        return (self.y0 + self.y1) / 2


@dataclass
class PageOCR:
    lines: List[Line]
    mean_conf: float
    psm: int
    columns_detected: bool
    word_count: int
    rotation: int = 0


# --------------------------------------------------------------------------- text cleanup
_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff\u00ad"), None)
_QUOTES = {ord("\u2018"): "'", ord("\u2019"): "'", ord("\u201c"): '"', ord("\u201d"): '"'}


def clean_text(text: str) -> str:
    """Normalise unicode quirks (ligatures, odd spaces, smart quotes)."""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_ZERO_WIDTH).translate(_QUOTES)
    return re.sub(r"[ \t\u00a0]+", " ", text).strip()


# --------------------------------------------------------------------------- tesseract
def _tesseract(settings: Settings):
    try:
        import pytesseract
    except ImportError as exc:  # pragma: no cover
        raise OCRError("pytesseract is not installed.") from exc
    if settings.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
    return pytesseract


def _run_tesseract(img: np.ndarray, psm: int, settings: Settings) -> List[Word]:
    pt = _tesseract(settings)
    try:
        d = pt.image_to_data(
            img, lang=settings.lang, config=f"--oem 3 --psm {psm}",
            output_type=pt.Output.DICT,
        )
    except pt.TesseractNotFoundError as exc:
        raise OCRError(
            "Tesseract executable not found. Install it and set TESSERACT_CMD "
            "(see README)."
        ) from exc
    except pt.TesseractError as exc:
        raise OCRError(f"Tesseract failed: {exc}") from exc

    words: List[Word] = []
    for i, raw in enumerate(d["text"]):
        txt = clean_text(raw or "")
        try:
            conf = float(d["conf"][i])
        except (TypeError, ValueError):
            conf = -1
        if not txt or conf < settings.min_word_conf:
            continue
        x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
        if w <= 0 or h <= 0:
            continue
        words.append(Word(txt, x, y, x + w, y + h, conf))
    return words


# --------------------------------------------------------------------------- layout
def detect_gutter(words: List[Word], page_w: int) -> Optional[tuple[int, int]]:
    """Find a vertical empty band separating two text columns.

    Returns (gutter_left, gutter_right) in pixels or None for single-column pages.
    The test counts text *rows* (not words) that cross a candidate band: a full-width
    header is only a few rows, whereas in a single-column page nearly every row crosses
    the centre of the page.
    """
    if len(words) < 40:
        return None
    rows = _group_rows(words)
    if len(rows) < 8:
        return None
    bin_w = max(4, page_w // 400)
    n_bins = page_w // bin_w + 1
    cover = np.zeros(n_bins, dtype=np.int32)
    for r in rows:
        mask = np.zeros(n_bins, dtype=bool)
        for w in r:
            mask[w.x0 // bin_w: w.x1 // bin_w + 1] = True
        cover += mask

    lo, hi = int(0.18 * n_bins), int(0.82 * n_bins)
    tol = max(2, int(0.15 * len(rows)))
    best, run_start = None, None
    for i in range(lo, hi + 1):
        empty = i < hi and cover[i] <= tol
        if empty and run_start is None:
            run_start = i
        elif not empty and run_start is not None:
            if best is None or (i - run_start) > (best[1] - best[0]):
                best = (run_start, i)
            run_start = None
    if best is None:
        return None
    g0, g1 = best[0] * bin_w, best[1] * bin_w
    if (g1 - g0) < 0.015 * page_w:
        return None

    # each side must hold a real column ...
    mid = (g0 + g1) / 2
    left = [w for w in words if (w.x0 + w.x1) / 2 < mid and w.x1 <= g0 + bin_w]
    right = [w for w in words if (w.x0 + w.x1) / 2 >= mid and w.x0 >= g1 - bin_w]
    if len(left) < 0.12 * len(words) or len(right) < 0.12 * len(words):
        return None
    # ... and the two sides must overlap vertically (otherwise it is just indentation)
    ly = (min(w.y0 for w in left), max(w.y1 for w in left))
    ry = (min(w.y0 for w in right), max(w.y1 for w in right))
    overlap = min(ly[1], ry[1]) - max(ly[0], ry[0])
    if overlap < 0.3 * min(ly[1] - ly[0], ry[1] - ry[0]):
        return None
    return g0, g1


def _group_rows(words: List[Word]) -> List[List[Word]]:
    """Group words into text rows by vertical overlap."""
    rows: List[dict] = []
    for w in sorted(words, key=lambda w: (w.yc, w.x0)):
        placed = False
        if rows:
            r = rows[-1]
            overlap = min(w.y1, r["y1"]) - max(w.y0, r["y0"])
            if overlap >= 0.5 * min(w.h, r["y1"] - r["y0"]):
                r["words"].append(w)
                n = len(r["words"])
                r["y0"] = (r["y0"] * (n - 1) + w.y0) / n
                r["y1"] = (r["y1"] * (n - 1) + w.y1) / n
                placed = True
        if not placed:
            rows.append({"words": [w], "y0": w.y0, "y1": w.y1})
    return [sorted(r["words"], key=lambda w: w.x0) for r in rows]


def _rows_to_lines(rows: List[List[Word]], page: int, insert_blanks: bool = True) -> List[Line]:
    if not rows:
        return []
    heights = [statistics.median(w.h for w in r) for r in rows]
    med_h = statistics.median(heights)
    out: List[Line] = []
    prev_bottom: Optional[float] = None
    for r, h in zip(rows, heights):
        top = min(w.y0 for w in r)
        bottom = max(w.y1 for w in r)
        if insert_blanks and prev_bottom is not None and (top - prev_bottom) > 0.9 * med_h:
            out.append(Line("", page))
        parts = [r[0].text]
        for a, b in zip(r, r[1:]):
            gap = b.x0 - a.x1
            parts.append(("   " if gap > 1.5 * h else " ") + b.text)
        out.append(Line("".join(parts), page, height=float(h),
                        conf=sum(w.conf for w in r) / len(r)))
        prev_bottom = bottom
    return out


def words_to_lines(words: List[Word], page_w: int, page: int,
                   detect_columns: bool = True) -> tuple[List[Line], bool]:
    """Rebuild reading-ordered lines from word boxes. Returns (lines, two_columns).

    Two-column pages: rows that run across the page (name, contact line, a section title
    spanning both columns) stay in place as "full-width rows"; the text between them is
    read left column first, then right column.
    """
    if not words:
        return [], False
    gutter = detect_gutter(words, page_w) if detect_columns else None
    if gutter is None:
        return _rows_to_lines(_group_rows(words), page), False

    mid = (gutter[0] + gutter[1]) / 2
    rows = _group_rows(words)

    def row_gap(r: List[Word]) -> Optional[float]:
        l = [w.x1 for w in r if w.x1 <= mid]
        rt = [w.x0 for w in r if w.x0 >= mid]
        return (min(rt) - max(l)) if l and rt else None

    # typical gap between the two columns, measured on rows that do not straddle the midline
    gaps = sorted(g for r in rows if not any(w.x0 < mid < w.x1 for w in r)
                  for g in [row_gap(r)] if g is not None)
    col_gap = gaps[int(0.2 * (len(gaps) - 1))] if gaps else (gutter[1] - gutter[0])

    full_idx, seg_of_row = [], []
    for i, r in enumerate(rows):
        straddles = any(w.x0 < mid < w.x1 for w in r)
        g = row_gap(r)
        if straddles or (g is not None and g < 0.5 * col_gap):
            full_idx.append(i)
    full_rows = [rows[i] for i in full_idx]
    full_y = [min(w.y0 for w in r) for r in full_rows]
    full_set = set(full_idx)

    segments: List[List[Word]] = [[] for _ in range(len(full_rows) + 1)]
    for i, r in enumerate(rows):
        if i in full_set:
            continue
        for w in r:
            segments[sum(1 for y in full_y if y < w.yc)].append(w)

    lines: List[Line] = []
    for i, seg in enumerate(segments):
        left = [w for w in seg if (w.x0 + w.x1) / 2 < mid]
        right = [w for w in seg if (w.x0 + w.x1) / 2 >= mid]
        for col in (left, right):
            if col:
                lines.extend(_rows_to_lines(_group_rows(col), page))
                lines.append(Line("", page))
        if i < len(full_rows):
            lines.extend(_rows_to_lines([full_rows[i]], page, insert_blanks=False))
    while lines and lines[-1].is_blank:
        lines.pop()
    return lines, True


# --------------------------------------------------------------------------- public API
def _mean_conf(words: List[Word]) -> float:
    return sum(w.conf for w in words) / len(words) if words else 0.0


def perform_ocr(image: np.ndarray, settings: Optional[Settings] = None,
                page: int = 0) -> PageOCR:
    """OCR one preprocessed page image into ordered lines.

    Strategy: run the primary PSM; if confidence is low, also try the fallback
    PSM and keep the better one. As a last resort (very low confidence) try
    90/180/270 degree rotations, which catches sideways phone scans.
    """
    settings = settings or Settings()
    h, w = image.shape[:2]

    best_words = _run_tesseract(image, settings.psm_primary, settings)
    best_psm, best_rot = settings.psm_primary, 0
    if _mean_conf(best_words) < settings.retry_below_conf:
        alt = _run_tesseract(image, settings.psm_fallback, settings)
        if _mean_conf(alt) > _mean_conf(best_words):
            best_words, best_psm = alt, settings.psm_fallback

    if _mean_conf(best_words) < settings.try_rotations_below_conf or len(best_words) < 5:
        for code, deg in ((cv2.ROTATE_90_CLOCKWISE, 90), (cv2.ROTATE_180, 180),
                          (cv2.ROTATE_90_COUNTERCLOCKWISE, 270)):
            rot = cv2.rotate(image, code)
            cand = _run_tesseract(rot, settings.psm_primary, settings)
            if len(cand) > len(best_words) and _mean_conf(cand) > _mean_conf(best_words) + 10:
                best_words, best_rot = cand, deg
                h, w = rot.shape[:2]

    lines, cols = words_to_lines(best_words, w, page, settings.detect_columns)
    return PageOCR(lines, _mean_conf(best_words), best_psm, cols, len(best_words), best_rot)


# ==============================================================================
# SECTIONS
# ==============================================================================
SECTION_SYNONYMS: Dict[str, List[str]] = {
    "summary": ["summary", "professional summary", "career summary", "executive summary",
                "career objective", "objective", "profile", "professional profile",
                "about me", "about", "personal statement", "summary of qualifications"],
    "education": ["education", "academic background", "educational qualifications",
                  "educational qualification", "academic qualifications", "academics",
                  "educational background", "qualifications", "academic profile",
                  "education details", "academic details", "education and training",
                  "education and qualifications", "academic record"],
    "experience": ["experience", "work experience", "professional experience",
                   "employment history", "work history", "career history", "employment",
                   "professional background", "industry experience", "relevant experience",
                   "work experience and internships", "experience and internships",
                   "employment experience", "career experience"],
    "internships": ["internships", "internship", "internship experience", "industrial training",
                    "internships and training", "summer internship", "internship details"],
    "projects": ["projects", "academic projects", "personal projects", "key projects",
                 "project work", "selected projects", "technical projects", "major projects",
                 "side projects", "project experience", "projects undertaken", "notable projects",
                 "relevant projects", "project details", "academic and personal projects"],
    "skills": ["skills", "technical skills", "key skills", "core skills", "skills and tools",
               "technical proficiency", "technical expertise", "tools and technologies",
               "core competencies", "areas of expertise", "technical summary", "it skills",
               "computer skills", "skills summary", "technical competencies", "skill set",
               "skillset", "technical skillset", "expertise", "proficiencies", "skills and abilities",
               "technical knowledge", "skills and technologies", "skills and expertise"],
    "certifications": ["certifications", "certification", "certificates", "licenses and certifications",
                       "certifications and licenses", "courses", "courses and certifications",
                       "training and certifications", "professional certifications", "online courses",
                       "licenses", "certifications and courses", "certifications and training",
                       "trainings and certifications", "certifications and achievements"],
    "achievements": ["achievements", "achievement", "awards", "honors", "honours", "awards and honors",
                     "achievements and awards", "accomplishments", "awards and recognition",
                     "honors and awards", "honours and awards", "scholarships", "recognition",
                     "key achievements", "awards and achievements", "awards and certifications"],
    "languages": ["languages known", "language proficiency", "spoken languages", "languages spoken"],
    "interests": ["interests", "hobbies", "hobbies and interests", "interests and hobbies",
                  "extracurricular activities", "extracurricular", "co curricular activities",
                  "activities", "volunteer experience", "volunteering", "volunteer work",
                  "positions of responsibility", "leadership and activities", "leadership"],
    "references": ["references", "reference"],
    "publications": ["publications", "research publications", "papers", "research"],
    "contact": ["contact", "contact me", "contact information", "contact details", "personal details",
                "personal information", "personal profile", "personal data"],
    "declaration": ["declaration"],
}
# Headings that are too generic to accept with inline content ("Research: ..." etc.)
_NO_INLINE = {"about", "research", "papers", "activities", "leadership", "employment", "expertise",
              "recognition", "objective", "profile", "qualifications"}
_BODY_SECTIONS = {"experience", "projects", "internships"}


def _norm(s: str) -> str:
    s = s.lower().replace("&", " and ").replace("/", " ")
    s = re.sub(r"[^a-z ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


_OCR_FIX = str.maketrans({"0": "o", "1": "i", "5": "s", "@": "o", "$": "s", "!": "i", "|": "l", "8": "b", "3": "e", "4": "a"})


def _norm_ocr(s: str) -> str:
    """Like _norm but first maps look-alike symbols to letters (C@NTACT -> contact)."""
    s = s.lower().translate(_OCR_FIX).replace("&", " and ").replace("/", " ")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", s)).strip()


_SYN_LOOKUP = {_norm(syn): sec for sec, syns in SECTION_SYNONYMS.items() for syn in syns}
_SYN_KEYS = list(_SYN_LOOKUP)
_LEAD_JUNK = re.compile(r"^[\W_\d]{0,4}\s*(?=[A-Za-z])")   # "01. ", "▌ ", "• "


def _match_heading(text: str, allow_fuzzy: bool = True) -> Optional[str]:
    t = _LEAD_JUNK.sub("", text.strip()).rstrip(":-–— ").strip()
    if not t or len(t.split()) > 6 or len(t) > 45:
        return None
    if re.search(r"[.!?;]|@\w+\.|https?:|\d{3,}", t):
        return None
    n = _norm(t)
    if not n:
        return None
    if n in _SYN_LOOKUP:
        return _SYN_LOOKUP[n]
    if allow_fuzzy and len(n) >= 6 and len(n.split()) <= 4:
        close = difflib.get_close_matches(n, _SYN_KEYS, n=1, cutoff=0.86)
        if close:
            return _SYN_LOOKUP[close[0]]
    # symbols that OCR substitutes for letters; only when the line has a symbol/digit to explain it
    if allow_fuzzy and re.search(r"[\d@$!|]", t):
        n2 = _norm_ocr(t)
        if n2 in _SYN_LOOKUP:
            return _SYN_LOOKUP[n2]
        close = difflib.get_close_matches(n2, _SYN_KEYS, n=1, cutoff=0.86)
        if close and len(n2) >= 6:
            return _SYN_LOOKUP[close[0]]
    return None


_INLINE_RE = re.compile(r"^(?P<h>[^:]{2,40}?)\s*:\s*(?P<rest>\S.*)$")


def detect_heading(text: str, current: str, next_text: str = "") -> tuple[Optional[str], str]:
    """Return (section_name, inline_content). section_name None => not a heading."""
    t = text.strip()
    if not t:
        return None, ""
    n = _norm(_LEAD_JUNK.sub("", t).rstrip(":-–— "))

    # bare "Languages": spoken-languages heading unless it is a sub-label inside Skills
    if n in {"languages", "language"}:
        nxt = _norm(next_text).split()
        if current != "skills" or (nxt and nxt[0] in SPOKEN_LANGUAGES):
            return "languages", ""
        return None, ""

    sec = _match_heading(t)
    if sec:
        return sec, ""

    # inline heading: "Skills: Python, Java" - not allowed inside body sections, where
    # labels like "Technologies:" or "Skills:" belong to the current entry
    if current not in _BODY_SECTIONS:
        m = _INLINE_RE.match(_LEAD_JUNK.sub("", t))
        if m:
            sec = _match_heading(m.group("h"), allow_fuzzy=False)
            if sec and _norm(m.group("h")) not in _NO_INLINE:
                return sec, m.group("rest").strip()
    return None, ""


def segment_sections(lines: List[Line]) -> Dict[str, List[Line]]:
    """Group lines by section. Lines before the first heading go to 'header'.

    Repeated headings (e.g. two-column or multi-page resumes that restate a
    section) are merged into the same key, preserving order.
    """
    sections: Dict[str, List[Line]] = {"header": []}
    current = "header"
    for i, ln in enumerate(lines):
        nxt = next((l.text for l in lines[i + 1:i + 4] if not l.is_blank), "")
        sec, inline = detect_heading(ln.text, current, nxt)
        if sec:
            current = sec
            sections.setdefault(current, [])
            if inline:
                sections[current].append(Line(inline, ln.page, ln.height, ln.conf))
            continue
        sections.setdefault(current, []).append(ln)
    return sections


# ==============================================================================
# PERSONAL
# ==============================================================================
_NOT_NAME_WORDS = {
    "resume", "curriculum", "vitae", "cv", "profile", "summary", "objective", "contact",
    "email", "phone", "mobile", "address", "linkedin", "github", "portfolio", "education",
    "experience", "skills", "projects", "engineer", "developer", "student", "analyst", "manager",
    "scientist", "intern", "designer", "consultant", "architect", "specialist", "fresher",
    "software", "data", "full", "stack", "frontend", "backend", "web", "machine", "learning",
    "senior", "junior", "lead", "associate", "technical", "professional", "career", "about",
    "me", "details", "personal", "information", "page", "of", "and", "the", "for", "ai", "ml",
    "bachelor", "master", "technology", "university", "college", "institute", "school",
    "tel", "ph", "dob", "gender", "nationality", "location", "website", "references",
}
_NAME_TOKEN = re.compile(r"^[A-Za-z][A-Za-z.'\-]*$")

KNOWN_CITIES = [
    "Bengaluru", "Bangalore", "Mumbai", "Delhi", "New Delhi", "Hyderabad", "Chennai", "Pune",
    "Kolkata", "Ahmedabad", "Noida", "Gurgaon", "Gurugram", "Jaipur", "Kochi", "Coimbatore",
    "Indore", "Chandigarh", "Mysuru", "Mysore", "Lucknow", "Bhubaneswar", "Nagpur", "Surat",
    "Thiruvananthapuram", "Visakhapatnam", "Vadodara", "Mangaluru", "Mangalore", "London",
    "New York", "San Francisco", "Seattle", "Toronto", "Singapore", "Dubai", "Berlin", "Sydney",
    "Austin", "Boston", "Chicago", "Los Angeles", "San Jose", "Dallas", "Atlanta", "Amsterdam",
    "Dhaka", "Karachi", "Colombo", "Kathmandu", "Melbourne", "Vancouver", "Paris", "Dublin",
]
_CITY_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, KNOWN_CITIES), key=len, reverse=True)) + r")\b")
_CITY_REGION_RE = re.compile(
    r"\b([A-Z][a-zA-Z.\-]+(?: [A-Z][a-zA-Z.\-]+){0,2}),\s*"
    r"([A-Z][a-zA-Z.\-]+(?: [A-Z][a-zA-Z.\-]+){0,2})(?:,\s*([A-Z][a-zA-Z.\-]+(?: [A-Z][a-zA-Z.\-]+)?))?\b")
_LABELLED_LOC = re.compile(r"(?:location|address|city|based in|residence)\s*[:\-]\s*(.+)", re.I)
_BAD_LOC_WORDS = re.compile(
    r"\b(university|college|institute|school|engineer|developer|technolog|solutions|pvt|ltd|inc|"
    r"limited|corp|bachelor|master|science|computer|analyst|manager|intern|software|systems)\b", re.I)


def _plausible_name(text: str) -> Optional[str]:
    t = re.sub(r"\s+", " ", text).strip(" ,;:|-–—")
    toks = t.split()
    if not (2 <= len(toks) <= 4):
        return None
    if re.search(r"[\d@/\\()_=+<>{}\[\]]", t):
        return None
    if any(not _NAME_TOKEN.match(tok) for tok in toks):
        return None
    if any(tok.lower().strip(".") in _NOT_NAME_WORDS for tok in toks):
        return None
    if all(len(tok.strip(".")) == 1 for tok in toks):
        return None
    if not all(tok[0].isupper() for tok in toks if tok.lower() not in {"van", "von", "de", "da", "bin", "al"}):
        return None
    if t.isupper() or t.islower():
        t = " ".join(w.capitalize() if len(w) > 1 else w for w in t.split())
    return t


def extract_name(header_lines: List[Line]) -> Optional[str]:
    """Pick the most name-like line near the top; prefer larger fonts (OCR line height)."""
    cands = []
    nonblank = [l for l in header_lines if not l.is_blank][:14]
    heights = [l.height for l in nonblank if l.height]
    med_h = sorted(heights)[len(heights) // 2] if heights else None
    for pos, ln in enumerate(nonblank):
        pieces = [ln.text] + split_segments(ln.text)
        seen = set()
        for piece in pieces:
            name = _plausible_name(piece)
            if not name or name in seen:
                continue
            seen.add(name)
            score = 10.0 - pos * 1.2
            if piece is not ln.text:
                score -= 0.5
            if med_h and ln.height:
                score += min(4.0, 3.0 * (ln.height / med_h - 1.0))
            cands.append((score, name))
    return max(cands)[1] if cands else None


def extract_name_by_font(all_lines: List[Line]) -> Optional[str]:
    """Fallback for layouts where the name is not in the top block (e.g. two-column with a
    sidebar read first): the name is usually the largest text on the page."""
    heights = sorted(l.height for l in all_lines if l.height and not l.is_blank)
    if len(heights) < 5:
        return None
    med = heights[len(heights) // 2]
    best = None
    for ln in all_lines:
        if not ln.height or ln.height < 1.35 * med:
            continue
        name = _plausible_name(ln.text)
        if name and (best is None or ln.height > best[0]):
            best = (ln.height, name)
    return best[1] if best else None


def extract_location(header_lines: List[Line]) -> Optional[str]:
    texts = [l.text for l in header_lines if not l.is_blank][:14]
    for t in texts:
        m = _LABELLED_LOC.search(t)
        if m:
            seg = split_segments(m.group(1))[0] if split_segments(m.group(1)) else m.group(1)
            if seg and not re.search(r"@|https?://", seg):
                return seg.strip()
    for t in texts:
        for seg in split_segments(EMAIL_RE.sub(" ", t)):
            if re.search(r"\d{4,}|@|https?://|www\.|linkedin|github", seg, re.I):
                # keep pin codes only if the rest looks like an address; otherwise skip
                seg = re.sub(r"\b\d{5,6}\b", "", seg).strip(" ,")
                if re.search(r"\d{3,}|@|https?://|www\.", seg):
                    continue
            if _BAD_LOC_WORDS.search(seg):
                continue
            m = _CITY_REGION_RE.search(seg)
            if m and len(seg) <= 60:
                return m.group(0)
            c = _CITY_RE.search(seg)
            if c and len(seg) <= 40:
                return seg.strip()
    return None


def extract_personal_information(header_lines: List[Line], full_text: str,
                                 all_lines: Optional[List[Line]] = None) -> Dict[str, Optional[str]]:
    """Contact details live in the header/contact sidebar; scan it first, then the full text."""
    head_text = "\n".join(l.text for l in header_lines)
    head_text_flat = head_text  # keep line structure for regexes that must not span lines

    def first(finder, *texts):
        for t in texts:
            v = finder(t)
            if v:
                return v
        return None

    emails = find_emails(head_text) or find_emails(full_text)
    phones = first(lambda t: (find_phones(t) or [None])[0], head_text, full_text)
    return {
        "full_name": extract_name(header_lines) or (extract_name_by_font(all_lines) if all_lines else None),
        "email": emails[0] if emails else None,
        "phone": phones,
        "location": extract_location(header_lines),
        "linkedin_url": first(find_linkedin, head_text_flat, full_text),
        "github_url": first(find_github_profile, head_text_flat, full_text),
        "portfolio_url": find_portfolio(head_text_flat),
    }


# ==============================================================================
# EDUCATION
# ==============================================================================
# --- degree patterns -----------------------------------------------------------------
_FIELDS = (r"Science|Technology|Engineering|Arts|Commerce|Business Administration|"
           r"Computer Applications|Computer Science|Fine Arts|Education|Laws|Medicine|Philosophy|"
           r"Management|Architecture|Pharmacy|Applied Science|Information Technology")
_DEGREE_PATTERNS = [
    # long forms
    rf"(?:Bachelor|Master|Doctor)(?:s)?(?:'s)?(?:\s+Degree)?\s+(?:of|in)\s+(?:the\s+)?(?:{_FIELDS})",
    r"(?:Bachelor|Master)(?:s)?(?:'s)?(?:\s+Degree)?\s+(?:of|in)\s+[A-Z][A-Za-z&]+(?:\s+[A-Z][A-Za-z&]+)?",
    r"Doctor(?:ate)?\s+of\s+Philosophy|Ph\.?\s?D\.?|Doctorate",
    r"Post[\s\-]?Graduate\s+Diploma(?:\s+in)?|Post[\s\-]?Graduation|Under[\s\-]?Graduate|Associate(?:'s)?\s+Degree|Diploma",
    # abbreviations (case-sensitive on purpose: "BE"/"MS"/"ME" are common English/other tokens)
    r"B\.?\s?Tech\.?|M\.?\s?Tech\.?|B\.\s?E\.?|M\.\s?E\.|B\.?\s?S[Cc]\.?|M\.?\s?S[Cc]\.?|B\.?\s?C\.?\s?A\.?|M\.?\s?C\.?\s?A\.?"
    r"|B\.?\s?B\.?\s?A\.?|M\.?\s?B\.?\s?A\.?|B\.?\s?Com\.?|M\.?\s?Com\.?|B\.?\s?A\.|M\.?\s?A\.|B\.\s?S\.|M\.\s?S\.|BS|MS|BE|ME|B\.?Des\.?|LL\.?B\.?|LL\.?M\.?|MBBS|B\.?Pharm\.?|M\.?Pharm\.?|BTech|MTech",
    # school level
    r"(?:Higher\s+Secondary(?!\s+School)|Senior\s+Secondary(?!\s+School)|Intermediate|Pre[\s\-]?University|"
    r"Class\s+(?:X{1,2}I{0,2}|1[02]|[IVX]+)|(?:10|12)th(?:\s+Grade|\s+Standard)?|(?:Grade|Std\.?|Standard)\s*(?:10|12)|SSLC|SSC|HSC|AISSCE|AISSE|PUC|CBSE|ICSE|ISC)",
]
_AMBIG_ABBR = {"BS", "MS", "BE", "ME", "BA", "MA"}
_AFTER_BAD = re.compile(r"^\s*(?:Office|Excel|Word|Access|Project|Teams|Visio|Paint|SQL|Azure)\b", re.I)
_DEGREE_RES = [re.compile(p) for p in _DEGREE_PATTERNS[:3]] + [
    re.compile(_DEGREE_PATTERNS[3], re.I),
    re.compile(r"(?<![A-Za-z.])(?:" + _DEGREE_PATTERNS[4] + r")(?![A-Za-z])"),
    re.compile(r"(?<![A-Za-z0-9])(?:" + _DEGREE_PATTERNS[5] + r")(?![A-Za-z0-9])", re.I),
]
_INST_RE = re.compile(
    r"\b(University|College|Institute|Institution|School|Academy|Polytechnic|Vidyalaya|Vidyapeeth|"
    r"Mahavidyalaya|IIT|NIT|IIIT|BITS|IISc|VIT|SRM|Campus)\b")
_SPEC_STOP = re.compile(
    r"\s[–—\-]\s|\b(?:from|at|with|CGPA|GPA|CPI|SGPA|Percentage|Grade|Aggregate|Expected|Graduat\w*|Batch|"
    r"Pursuing|Completed|Class of|Score|Marks)\b|\d{4}|\d+(?:\.\d+)?\s*%|[,|•(]", re.I)


def find_degree(text: str) -> Optional[tuple[str, int, int]]:
    """Leftmost degree token in text: (token, start, end)."""
    best = None
    for rx in _DEGREE_RES:
        for m in rx.finditer(text):
            tok = m.group(0).strip()
            if tok in _AMBIG_ABBR and _AFTER_BAD.match(text[m.end():]):
                continue
            # ambiguous 2-letter caps only count when followed by a delimiter (BE in / BE, / BE -)
            if tok in _AMBIG_ABBR and not re.match(r"\s*(?:in\b|,|\(|-|–|/|\||$)", text[m.end():]):
                continue
            if best is None or m.start() < best[1] or (m.start() == best[1] and m.end() > best[2]):
                best = (tok, m.start(), m.end())
            break
    return best


def _clean_spec(rest: str) -> Optional[str]:
    m = re.match(r"\s*\(\s*([^)]+?)\s*\)", rest)
    if m:
        spec = m.group(1)
    else:
        m = re.match(r"\s*(?:in|of|-|–|—|:|,|/|\()\s*(.+)", rest, re.I)
        if not m:
            return None
        spec = m.group(1)
    stop = _SPEC_STOP.search(spec)
    if stop:
        spec = spec[:stop.start()]
    # drop a trailing institution name glued onto the specialization
    inst = _INST_RE.search(spec)
    if inst:
        return None
    spec = spec.strip(" ,.-–—:|")
    if not spec or len(spec) > 60 or len(spec.split()) > 7:
        return None
    return spec


_INDIAN_INST = re.compile(r"^(?:Indian|National|International|Birla|Visvesvaraya)\b.*Institute of "
                          r"(?:Technology|Management|Science|Information Technology)$")


def _institution(lines: List[str]) -> Optional[str]:
    """Institution = the comma/pipe-separated piece of a line that holds an institution keyword."""
    for line in lines:
        if is_bullet(line) and len(line) > 90:
            continue
        cleaned = strip_date_range(line)
        cleaned = re.sub(r"(?:CGPA|GPA|CPI|Percentage|Aggregate)\b.*$", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\b\d{2,3}(?:\.\d+)?\s*%", " ", cleaned)
        cleaned = re.sub(r"(?<!\d)(?:19|20)\d{2}(?!\d)", " ", cleaned)
        segs = [s.strip(" ,-–—") for part in split_segments(cleaned, split_at=False) for s in part.split(",") if s.strip(" ,-–—")]
        for i, seg in enumerate(segs):
            if _INST_RE.search(seg) and len(seg) < 90:
                name = seg
                dg = find_degree(name)
                if dg and dg[1] == 0:        # "B.Tech XYZ University": drop the degree prefix
                    tail = name[dg[2]:]
                    m = _INST_RE.search(tail)
                    if m:
                        words = tail[:m.end()].split()
                        j = len(words) - 1
                        while j > 0 and (words[j - 1][:1].isupper() or words[j - 1].lower() in {"of", "and", "&"}):
                            j -= 1
                        name = " ".join(words[j:])
                nxt = segs[i + 1] if i + 1 < len(segs) else ""
                if _INDIAN_INST.match(name) and re.fullmatch(r"[A-Z][a-z]+", nxt):
                    name = f"{name}, {nxt}"        # "Indian Institute of Technology, Bombay"
                return name.strip(" ,-–—()")
    return None


def _graduation_year(block_text: str) -> Optional[str]:
    now = date.today().year
    r = find_date_range(block_text)
    if r:
        end = r[1]
        return end[:4] if end != "Present" else None
    years = [int(y) for y in YEAR_RE.findall(block_text) if 1970 <= int(y) <= now + 8]
    return str(years[-1]) if years else None


def _split_blocks(lines: List[str]) -> List[List[str]]:
    blocks: List[List[str]] = []
    cur: List[str] = []
    has_deg = has_inst = False
    for line in lines:
        if not line.strip():
            continue
        is_desc = is_bullet(line) or len(line) > 110
        deg = (not is_desc) and find_degree(line) is not None
        inst = (not is_desc) and bool(_INST_RE.search(line)) and len(line) < 100
        if cur and ((deg and has_deg) or (inst and has_inst)):
            blocks.append(cur)
            cur, has_deg, has_inst = [], False, False
        cur.append(line)
        has_deg |= deg
        has_inst |= inst
    if cur:
        blocks.append(cur)
    return blocks


def _parse_block(block: List[str]) -> Dict[str, Optional[str]]:
    text = " ".join(block)
    degree = spec = None
    for line in block:
        if is_bullet(line):
            continue
        found = find_degree(line)
        if found:
            tok, s, e = found
            degree = re.sub(r"\s+", " ", tok).strip()
            rest = line[e:]
            school = re.match(r"(?i)(?:higher|senior|secondary|high|intermediate|pre|class|\d|grade|std|ssc|hsc|puc|cbse|icse|isc|aiss)", degree)
            if school:
                pm = re.match(r"\s*\(\s*((?:Class|Grade|Std)[^)]*|X{1,2}I{0,2}|1[02])\s*\)", rest, re.I)
                if pm:
                    degree = f"{degree} ({pm.group(1).strip()})"
                    rest = rest[pm.end():]
                sm = re.search(r"\b(Science|Commerce|Arts|PCM[B]?|Humanities)\b", rest)
                spec = sm.group(1) if sm else None
            else:
                spec = _clean_spec(rest)
            # specialization may sit on the line before ("Computer Science" / "B.Tech")
            break
    if degree and not spec:
        m = re.search(r"(?:Specialization|Specialisation|Major|Branch|Stream)\s*[:\-]\s*([^,|\n(]+)", text, re.I)
        if m:
            spec = m.group(1).strip(" .")
    return {
        "institution_name": _institution(block),
        "degree": degree,
        "specialization": spec,
        "graduation_year": _graduation_year(text),
        "CGPA_or_percentage": find_grade(text),
        "source_text": text[:300],
    }


def extract_education(section_lines: List[str]) -> List[Dict[str, Optional[str]]]:
    """Parse the education section into entries.

    Blocks start when a second degree or a second institution appears, which handles both
    'Degree / Institution / dates' and 'Institution / Degree / dates' layouts.
    """
    entries = []
    for block in _split_blocks(section_lines):
        e = _parse_block(block)
        if e["degree"] or e["institution_name"]:
            entries.append(e)
    return entries


# ==============================================================================
# EXPERIENCE
# ==============================================================================
TITLE_WORDS = {
    "engineer", "developer", "intern", "internship", "manager", "analyst", "consultant", "scientist",
    "lead", "associate", "designer", "architect", "administrator", "specialist", "officer", "director",
    "executive", "trainee", "researcher", "assistant", "programmer", "tester", "sde", "head",
    "coordinator", "technician", "editor", "writer", "professor", "teacher", "founder", "co-founder",
    "president", "volunteer", "freelancer", "freelance", "apprentice", "fellow", "instructor", "tutor",
    "supervisor", "representative", "advisor", "mentor", "member", "student", "sde-1", "sde-2", "swe",
    "tech", "ambassador", "owner", "partner", "vp", "cto", "ceo", "strategist", "evangelist", "dba",
}
COMPANY_WORDS = {
    "inc", "inc.", "ltd", "ltd.", "llc", "pvt", "pvt.", "private", "limited", "corp", "corp.", "corporation",
    "technologies", "technology", "solutions", "systems", "labs", "services", "software", "consulting",
    "group", "bank", "university", "company", "studios", "industries", "enterprises", "infotech",
    "softwares", "networks", "global", "research", "institute", "foundation", "ventures", "analytics",
    "innovations", "digital", "co.", "gmbh", "plc", "college", "hospital", "agency", "partners",
}
_EMP_TYPE = re.compile(r"^(?:full[\s\-]?time|part[\s\-]?time|contract|permanent|remote|hybrid|on[\s\-]?site|"
                       r"temporary|internship)$", re.I)
_LOC_WORDS = {c.lower() for c in KNOWN_CITIES} | {
    "india", "usa", "uk", "u.s.", "canada", "germany", "remote", "karnataka", "maharashtra", "telangana",
    "tamil nadu", "delhi", "ca", "ny", "tx", "wa", "ma", "il", "singapore", "australia"}


# ------------------------------------------------------------------------- generic grouping
_LABEL_LINE = re.compile(
    r"^\s*(?:Tech(?:nolog\w*)?(?:\s+Stack)?(?:\s+Used)?|Tools?(?:\s+Used)?|Environment|Stack|Skills(?:\s+Used)?|"
    r"Responsibilit\w*|Key\s+\w+|Achievements?|Description|Role|Duration|Location|Client|Domain|Team\s+Size|"
    r"Built\s+(?:with|using)|Languages?|Libraries|Frameworks?|Link|URL|GitHub|Demo|Live|Source|Repo\w*|Impact|Result\w*)"
    r"\s*[:\-–—]\s*\S", re.I)
_HEADER_CUE = re.compile(r"\s[|\-–—:]\s|\(|\|")


def is_header_like(line: str) -> bool:
    """Short, capitalised, non-bullet line that does not read like a sentence or a label."""
    t = line.strip()
    if not t or is_bullet(t) or len(t) > 100 or t.endswith((".", ";", ",")) or _LABEL_LINE.match(t):
        return False
    return t[0].isupper()


def group_by_headers(lines: List[str], max_header_lines: int = 3) -> List[Dict[str, List[str]]]:
    """Split a section into entries of {'header': [...], 'body': [...]} without relying on dates."""
    nonblank = [l for l in lines if l.strip()]
    if not nonblank:
        return []
    max_len = max(len(l) for l in nonblank)
    entries: List[Dict[str, List[str]]] = []
    cur = {"header": [], "body": []}
    blank_before = True
    prev: Optional[str] = None
    for l in lines:
        if not l.strip():
            blank_before = True
            continue
        hl = is_header_like(l) and title_ratio(l) >= 0.5
        prev_long = prev is not None and len(prev) >= 0.9 * max_len
        prev_ends = prev is not None and (is_bullet(prev) or prev.rstrip().endswith("."))
        # a header right after body text counts as a new entry unless it looks like the
        # wrapped tail of a full-width previous line (no separator/parenthesis cue)
        starts_new = bool(cur["body"]) and hl and (
            blank_before or (prev_ends and (not prev_long or bool(_HEADER_CUE.search(l)))))
        if starts_new:
            entries.append(cur)
            cur = {"header": [], "body": []}
        if hl and not cur["body"] and len(cur["header"]) < max_header_lines:
            cur["header"].append(l)
        else:
            cur["body"].append(l)
        blank_before = False
        prev = l
    if cur["header"] or cur["body"]:
        entries.append(cur)
    return entries


# ------------------------------------------------------------------------- header parsing
def _is_location(seg: str) -> bool:
    s = seg.lower().strip(" .")
    if s in _LOC_WORDS:
        return True
    parts = [p.strip() for p in re.split(r"[,/]", s)]
    return bool(parts) and all(p in _LOC_WORDS for p in parts)


def _split_header_segments(text: str) -> List[str]:
    segs: List[str] = []
    for s in split_segments(text):
        pieces = [p.strip() for p in s.split(",")]
        merged: List[str] = []
        for p in pieces:
            if merged and re.fullmatch(r"(?:Inc|Ltd|LLC|Pvt|Limited|Corp|Co)\.?", p, re.I):
                merged[-1] += f", {p}"
            elif p:
                merged.append(p)
        segs.extend(merged)
    return [s for s in segs if not _EMP_TYPE.match(s) and not _is_location(s)]


def _score(seg: str, vocab: set) -> int:
    return sum(1 for w in re.findall(r"[A-Za-z][A-Za-z.\-]*", seg.lower()) if w in vocab)


def parse_header(header_lines: List[str]) -> tuple[Optional[str], Optional[str]]:
    """Return (job_title, company_name) from the header lines of one entry."""
    segs: List[str] = []
    for l in header_lines:
        segs.extend(_split_header_segments(strip_date_range(l)))
    segs = [s for s in segs if len(s) > 1 and not re.fullmatch(r"[\W\d]+", s)]
    if not segs:
        return None, None
    title = max(segs, key=lambda s: _score(s, TITLE_WORDS), default=None)
    if title is not None and _score(title, TITLE_WORDS) == 0:
        title = None
    rest = [s for s in segs if s != title]
    company = max(rest, key=lambda s: _score(s, COMPANY_WORDS), default=None) if rest else None
    if company is not None and _score(company, COMPANY_WORDS) == 0:
        company = next((s for s in rest if title_ratio(s) >= 0.5), None)
    return title, company


# ------------------------------------------------------------------------- main extractor
def _anchor_ok(line: str) -> bool:
    return bool(line.strip()) and not is_hard_bullet(line) and len(line) <= 140 and find_date_range(line) is not None


def extract_experience(section_lines: List[str]) -> List[Dict[str, Optional[str]]]:
    lines = list(section_lines)
    anchors = [i for i, l in enumerate(lines) if _anchor_ok(l)]
    if not anchors:
        return _from_headers(lines)

    entries = []
    consumed_until = -1
    headers: List[tuple[int, int]] = []   # (header_start, anchor_idx)
    for a in anchors:
        start = a
        j = a - 1
        while j > consumed_until and j >= 0 and a - j <= 3:
            l = lines[j]
            if not l.strip() or is_bullet(l) or j in anchors:
                break
            if not (is_header_like(l) and title_ratio(l) >= 0.5):
                break
            start = j
            j -= 1
        headers.append((start, a))
        consumed_until = a

    for k, (start, a) in enumerate(headers):
        end = headers[k + 1][0] if k + 1 < len(headers) else len(lines)
        header = lines[start:a + 1]
        body_idx = list(range(a + 1, end))
        title, company = parse_header(header)
        # company/title may sit on the lines right below the date line
        extra = 0
        while (title is None or company is None) and extra < 2 and body_idx:
            nxt = lines[body_idx[0]]
            if nxt.strip() and not is_bullet(nxt) and is_header_like(nxt) and title_ratio(nxt) >= 0.5 \
                    and find_date_range(nxt) is None:
                header = header + [nxt]
                body_idx.pop(0)
                title, company = parse_header(header)
                extra += 1
            else:
                break
        rng = find_date_range(lines[a])
        body = [lines[i] for i in body_idx]
        desc = "\n".join(merge_wrapped(body)).strip()
        entries.append({
            "company_name": company,
            "job_title": title,
            "start_date": rng[0] if rng else None,
            "end_date": rng[1] if rng else None,
            "description": desc or None,
        })
    return [e for e in entries if e["company_name"] or e["job_title"]]


def _from_headers(lines: List[str]) -> List[Dict[str, Optional[str]]]:
    out = []
    for e in group_by_headers(lines):
        if not e["header"]:
            continue
        title, company = parse_header(e["header"])
        if not (title or company):
            continue
        out.append({
            "company_name": company, "job_title": title, "start_date": None, "end_date": None,
            "description": "\n".join(merge_wrapped(e["body"])).strip() or None,
        })
    return out


def split_internships(entries: List[Dict[str, Optional[str]]]):
    """Separate entries whose title says intern/trainee from regular experience."""
    jobs, interns = [], []
    for e in entries:
        t = (e.get("job_title") or "") + " " + (e.get("company_name") or "")
        (interns if re.search(r"\b(intern|internship|trainee|apprentice)\b", t, re.I) else jobs).append(e)
    return jobs, interns


# ==============================================================================
# PROJECTS
# ==============================================================================
_TECH_LABEL = re.compile(
    r"^\s*(?:Tech(?:nologies|nology)?(?:\s+Stack)?(?:\s+Used)?|Tools?(?:\s+(?:Used|&\s*Technologies|and Technologies))?|"
    r"Built\s+(?:with|using)|Stack|Environment|Languages?(?:\s+Used)?|Libraries|Frameworks?|Skills(?:\s+Used)?)"
    r"\s*[:\-–—]\s*(.+)$", re.I)
_LINK_LABEL = re.compile(r"^\s*(?:link|url|github|demo|live|source|repo(?:sitory)?)\s*[:\-–]", re.I)
_NAME_SPLIT = re.compile(r"\s*[|•]\s*|\s+[-–—]\s+|\s*:\s+")


def parse_tech_list(text: str) -> List[str]:
    """'Python, Flask & PostgreSQL' -> ['Python', 'Flask', 'PostgreSQL'] (canonical names when known)."""
    text = re.sub(r"[()\[\]]", ",", text)
    out: List[str] = []
    for tok in re.split(r"[,;/|•·]+|\s+&\s+|\s+and\s+|\s+\+\s+", text):
        tok = tok.strip(" .-–—:")
        if not tok or len(tok) > 40 or len(tok.split()) > 4:
            continue
        canon = canonicalize(tok) or tok
        if canon.lower() not in {t.lower() for t in out}:
            out.append(canon)
    return out


def _looks_like_tech(text: str) -> bool:
    toks = [t for t in re.split(r"[,;/|&]+|\s+and\s+", re.sub(r"[()]", ",", text)) if t.strip()]
    if len(toks) < 1 or len(text) > 120:
        return False
    known = sum(1 for t in toks if canonicalize(t.strip(" .")))
    return known >= max(1, len(toks) // 2)


def _parse_header(header_lines: List[str]) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (name, technologies_text, description_text) from the project header line(s)."""
    first = strip_date_range(strip_bullet(header_lines[0]))
    first = _GENERIC_URL_RE.sub(" ", first)
    first = re.sub(r"\(\s*\)", " ", first)
    m = re.match(r"^(.*?)\s*\(([^()]*)\)\s*$", first)           # "Name (Python, Flask)"
    if m and _looks_like_tech(m.group(2)):
        return m.group(1).strip(" -–—|:"), m.group(2), None
    parts = _NAME_SPLIT.split(first, maxsplit=1)
    name = parts[0].strip(" -–—|:")
    rest = parts[1].strip() if len(parts) > 1 else ""
    tech = desc = None
    if rest:
        if _looks_like_tech(rest):
            tech = rest
        else:
            desc = rest
    extra = " ".join(header_lines[1:]).strip()
    if extra:
        if not tech and _looks_like_tech(extra):
            tech = extra
        else:
            desc = f"{desc} {extra}".strip() if desc else extra
    return name or None, tech, desc


def extract_projects(section_lines: List[str]) -> List[Dict[str, object]]:
    projects = []
    for e in group_by_headers(section_lines):
        if not e["header"]:
            continue
        name, tech_text, desc_head = _parse_header(e["header"])
        if not name or len(name) < 2 or re.fullmatch(r"[\W\d]+", name):
            continue
        tech: List[str] = parse_tech_list(tech_text) if tech_text else []
        desc_items: List[str] = [desc_head] if desc_head else []
        body_lines = []
        for bl in e["body"]:
            m = _TECH_LABEL.match(strip_bullet(bl))
            if m:                                   # "Technologies: ..." lines are never description
                for t in parse_tech_list(m.group(1)):
                    if t.lower() not in {x.lower() for x in tech}:
                        tech.append(t)
            else:
                body_lines.append(bl)
        for item in merge_wrapped(body_lines):
            if not _LINK_LABEL.match(item) and not re.fullmatch(_GENERIC_URL_RE, item.strip()):
                desc_items.append(item)
        desc = "\n".join(desc_items).strip() or None
        if not tech:  # no explicit stack: scan THIS project's own text only
            tech = scan_known_skills(" ".join([name, desc or ""]))
        projects.append({"project_name": name, "project_description": desc, "technologies_used": tech})
    return projects


# ==============================================================================
# ADDITIONAL
# ==============================================================================
_NOISE = re.compile(r"^(?:page\s*\d+|\d+\s*/\s*\d+|[\W_]+)$", re.I)


def extract_list_section(section_lines: List[str]) -> List[str]:
    """Each logical item (bullet or wrapped group) becomes one string."""
    items, seen = [], set()
    for it in merge_wrapped(section_lines):
        it = re.sub(r"\s+", " ", it).strip(" -–—•")
        if len(it) < 3 or _NOISE.match(it) or it.lower() in seen:
            continue
        seen.add(it.lower())
        items.append(it)
    return items


def extract_certifications(section_lines: List[str]) -> List[str]:
    return extract_list_section(section_lines)


def extract_achievements(section_lines: List[str]) -> List[str]:
    return extract_list_section(section_lines)


# ==============================================================================
# SKILLS
# ==============================================================================
_LABEL_RE = re.compile(r"^\s*([A-Za-z][A-Za-z &/+()\-]{1,45}?)\s*[:\-–]\s*(\S.*)$")
_LABEL_ONLY_RE = re.compile(r"^\s*([A-Za-z][A-Za-z &/+()\-]{1,45}?)\s*:\s*$")


_LABEL_WORDS = {"languages", "language", "programming languages", "programming", "tools", "frameworks",
                "framework", "libraries", "databases", "database", "technologies", "platforms", "skills",
                "cloud", "devops", "tools and technologies", "technical skills", "others", "other"}
_TRAIL_STOP = {"in", "of", "and", "the", "with", "for", "to", "at", "on"}
_NOT_SKILL = re.compile(r"\b(?:University|College|Institute|School|Bachelor|Master|CGPA|GPA|Percentage|"
                        r"B\.?Tech|M\.?Tech|B\.?E\.?|Engineering in|Present|Intern(?:ship)?)\b|\b(?:19|20)\d{2}\b")


def _label_category(label: str):
    l = label.lower()
    if re.search(r"language", l) and not re.search(r"natural|spoken|human", l):
        return PL
    if re.search(r"framework|librar|package|module", l):
        return FW
    if re.search(r"tool|technolog|platform|database|\bdb\b|cloud|devops|software|\bide\b|operating|\bos\b|"
                 r"version control|deployment|testing", l):
        return TOOLS
    if re.search(r"soft|interpersonal|personal|hobb|interest", l):
        return "ignore"
    return None  # generic ("Technical Skills", "Core Competencies") -> classify per token


def _split_tokens(text: str) -> List[str]:
    text = re.sub(r"\([^)]*\)", lambda m: "," + m.group(0)[1:-1] + ",", text)   # "Python (Pandas, NumPy)"
    # ". " is a common OCR misread of ", " (e.g. "Git. Docker. AWS")
    toks = re.split(r"[,;|•·●▪]+|\s{3,}|\s+[-–—]\s+|(?<=[a-z0-9])\.\s+(?=[A-Z])", text)
    out = []
    for t in toks:
        t = t.strip(" .:-–—*")
        if not t:
            continue
        if canonicalize(t):
            out.append(t)
        elif "/" in t and all(canonicalize(p) for p in t.split("/") if p.strip()):
            out.extend(p.strip() for p in t.split("/") if p.strip())   # "HTML/CSS/JS"
        elif " and " in t.lower() and all(canonicalize(p) for p in re.split(r"\s+and\s+", t, flags=re.I)):
            out.extend(re.split(r"\s+and\s+", t, flags=re.I))
        else:
            out.append(t)
    return out


def _valid_unknown(tok: str) -> bool:
    """Accept unknown tokens only if they look like a skill name, not a sentence."""
    if not (2 <= len(tok) <= 35) or len(tok.split()) > 4:
        return False
    if re.search(r"[.!?]$|\d{4}|@|https?:", tok):
        return False
    low = tok.lower().strip()
    if low in _LABEL_WORDS or _NOT_SKILL.search(tok) or low.split()[-1] in _TRAIL_STOP:
        return False
    if low in SOFT_SKILLS or low in SPOKEN_LANGUAGES:
        return False
    if re.fullmatch(r"[\d\W]+", tok):
        return False
    if re.match(r"^(?:basic|intermediate|advanced|beginner|expert|proficient|good|strong|knowledge|"
                r"familiar|experience|worked|developed|built|hands-on)\b", low):
        return False
    return True


def extract_skills(section_lines: List[str]) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {PL: [], FW: [], TOOLS: [], TECH: []}

    def add(cat: str, name: str):
        if name.lower() not in {x.lower() for x in result[cat]}:
            result[cat].append(name)

    pending_label_cat = None     # for "Languages:" on one line and values on the next
    pending_active = False
    for raw in section_lines:
        line = raw.strip()
        if not line:
            pending_active = False
            continue
        text = strip_bullet(line) if is_bullet(line) else line
        label_cat, has_label, body = None, False, text

        if text.lower().strip(" :") in _LABEL_WORDS:          # sidebar style: label on its own line
            pending_label_cat, pending_active = _label_category(text.strip(" :")), True
            continue
        m = _LABEL_RE.match(text)
        m_only = _LABEL_ONLY_RE.match(text)
        if m and not canonicalize(m.group(1)):
            has_label, label_cat, body = True, _label_category(m.group(1)), m.group(2)
        elif m_only and not canonicalize(m_only.group(1)):
            pending_label_cat, pending_active = _label_category(m_only.group(1)), True
            continue
        elif pending_active:
            label_cat, has_label = pending_label_cat, True

        if has_label and label_cat == "ignore":
            continue
        for tok in _split_tokens(body):
            canon = canonicalize_fuzzy(tok)
            if canon:
                add(category_of(canon), canon)   # known terms: the vocabulary decides the bucket
            elif _valid_unknown(tok):
                add(label_cat or TECH, tok.strip())
    return result


# ==============================================================================
# VALIDATION
# ==============================================================================
CORE_FIELDS = ("full_name", "contact", "education_or_experience", "skills")


def _missing_fields(d: Dict[str, Any]) -> List[str]:
    missing = [f"personal_information.{k}" for k, v in d["personal_information"].items() if not v]
    missing += [f"skills.{k}" for k, v in d["skills"].items() if not v]
    missing += [k for k in ("education", "experience", "projects", "certifications",
                            "achievements", "internships") if not d[k]]
    return missing


def validate_resume(d: Dict[str, Any], ocr_confidence: float | None = None,
                    min_text_chars: int = 40, good_ocr_conf: float = 70.0,
                    section_sizes: Dict[str, int] | None = None) -> Dict[str, Any]:
    """Mutates and returns `d` with warnings, missing_fields and extraction_status.

    Status meaning:
      failed        - no usable text
      success       - all core fields present, no warnings, OCR confidence high
      needs_review  - anything else (the safe default; a human should look)
    'success' is NOT a claim of correctness, only that no automatic check failed.
    """
    w: List[str] = []
    p = d["personal_information"]
    text = d.get("raw_text") or ""

    if len(text.strip()) < min_text_chars:
        d["warnings"] = ["Almost no text was extracted (blank, unreadable or non-resume document)."]
        d["missing_fields"] = _missing_fields(d)
        d["extraction_status"] = "failed"
        return d

    if ocr_confidence is not None and ocr_confidence < good_ocr_conf:
        w.append(f"Low OCR confidence ({ocr_confidence:.0f}/100): verify all fields.")

    if p["email"] and not _EMAIL_STRICT.match(p["email"]):
        w.append("personal_information.email: invalid format")
    if p["phone"]:
        digits = re.sub(r"\D", "", p["phone"])
        if not 10 <= len(digits) <= 15:
            w.append("personal_information.phone: unexpected digit count")
    for k in ("linkedin_url", "github_url", "portfolio_url"):
        if p[k] and not re.match(r"^https?://[^\s/]+\.[A-Za-z]{2,}", p[k]):
            w.append(f"personal_information.{k}: malformed URL")
    if p["full_name"] and len(p["full_name"].split()) < 2:
        w.append("personal_information.full_name: only one word - verify")

    now = date.today().year
    for i, e in enumerate(d["education"]):
        if not e.get("institution_name"):
            w.append(f"education[{i}]: institution not found")
        if not e.get("degree"):
            w.append(f"education[{i}]: degree not found")
        gy = e.get("graduation_year")
        if gy and not (1970 <= int(gy) <= now + 8):
            w.append(f"education[{i}]: implausible graduation_year {gy}")
        g = e.get("CGPA_or_percentage")
        if g and "/" not in g and "%" not in g:
            w.append(f"education[{i}]: grade '{g}' has no scale; confirm CGPA out of 10 or 4")

    for section in ("experience", "internships"):
        for i, e in enumerate(d[section]):
            if not e.get("company_name"):
                w.append(f"{section}[{i}]: company not found")
            if not e.get("job_title"):
                w.append(f"{section}[{i}]: job title not found")
            s, en = date_sort_key(e.get("start_date")), date_sort_key(e.get("end_date"))
            if s and en and s > en:
                w.append(f"{section}[{i}]: start_date is after end_date")
            if not e.get("start_date"):
                w.append(f"{section}[{i}]: no date range found")

    for i, pr in enumerate(d["projects"]):
        if not pr.get("project_description") and not pr.get("technologies_used"):
            w.append(f"projects[{i}]: name only - description/technologies missing")

    # a heading was recognised and had content, but nothing was parsed from it
    parsed = {"education": d["education"], "experience": d["experience"] + d["internships"],
              "projects": d["projects"], "skills": [x for s in d["skills"].values() for x in s],
              "certifications": d["certifications"], "achievements": d["achievements"],
              "internships": d["internships"]}
    for sec, size in (section_sizes or {}).items():
        if sec in parsed and size >= 2 and not parsed[sec]:
            w.append(f"section '{sec}' found ({size} lines) but no entries were parsed - review")

    core_ok = (
        bool(p["full_name"]) and bool(p["email"] or p["phone"])
        and bool(d["education"] or d["experience"] or d["internships"])
        and any(d["skills"].values())
    )
    if not p["full_name"]:
        w.append("full_name not found")
    if not (p["email"] or p["phone"]):
        w.append("no email or phone found")
    if not (d["education"] or d["experience"] or d["internships"]):
        w.append("no education or experience entries found")
    if not any(d["skills"].values()):
        w.append("no skills found")

    d["warnings"] = w
    d["missing_fields"] = _missing_fields(d)
    d["extraction_status"] = "success" if core_ok and not w else "needs_review"
    return d


# ==============================================================================
# PIPELINE
# ==============================================================================
log = logging.getLogger(__name__)

_SKILL_LABEL_LINE = re.compile(
    r"^\s*(?:Programming\s+)?(?:Languages?|Frameworks?|Libraries|Tools?|Technologies|Databases?|Platforms|Cloud)"
    r"[^:]{0,25}:\s*\S", re.I)


def empty_result() -> Dict[str, Any]:
    return {
        "personal_information": {k: None for k in (
            "full_name", "email", "phone", "location", "linkedin_url", "github_url", "portfolio_url")},
        "education": [],
        "skills": {"technical_skills": [], "programming_languages": [],
                   "frameworks_and_libraries": [], "tools_and_technologies": []},
        "experience": [],
        "projects": [],
        "certifications": [],
        "achievements": [],
        "internships": [],
        "extraction_status": "needs_review",
        "missing_fields": [],
        "raw_text": "",
        # additive debug fields (not part of the core schema; ignore or strip if unwanted)
        "warnings": [],
        "ocr_confidence": None,
        "page_count": 0,
    }


def _texts(sections: Dict[str, List[Line]], key: str) -> List[str]:
    return [l.text for l in sections.get(key, [])]


def extract_from_lines(lines: List[Line], ocr_confidence: Optional[float] = None,
                       settings: Optional[Settings] = None) -> Dict[str, Any]:
    """Stages 6-9 on already-extracted text lines (no OCR involved)."""
    settings = settings or Settings()
    result = empty_result()
    result["raw_text"] = "\n".join(l.text for l in lines)
    result["ocr_confidence"] = None if ocr_confidence is None else round(ocr_confidence, 1)

    sections = segment_sections(lines)
    contact_lines = sections.get("header", []) + sections.get("contact", [])
    full_text = result["raw_text"]

    result["personal_information"] = extract_personal_information(contact_lines, full_text, lines)
    notes: List[str] = []
    result["education"] = extract_education(_texts(sections, "education"))
    if not result["education"]:
        # heading lost to OCR (common on dark sidebars): accept only entries with BOTH a degree
        # and an institution found outside the sections that legitimately contain such words
        loose = [l.text for k, v in sections.items()
                 if k not in ("experience", "projects", "internships", "certifications", "achievements")
                 for l in v]
        rec = [e for e in extract_education(loose) if e["degree"] and e["institution_name"]]
        if rec:
            result["education"] = rec
            notes.append("Education was recovered without a recognisable heading - verify.")
    skill_lines = _texts(sections, "skills")
    result["skills"] = extract_skills(skill_lines)
    if not any(result["skills"].values()):
        rec_lines = [l.text for v in sections.values() for l in v if _SKILL_LABEL_LINE.match(l.text)]
        if rec_lines:
            result["skills"] = extract_skills(rec_lines)
            notes.append("Skills were recovered from labelled lines without a skills heading - verify.")
    jobs, interns_from_exp = split_internships(extract_experience(_texts(sections, "experience")))
    result["experience"] = jobs
    result["internships"] = (extract_experience(_texts(sections, "internships")) + interns_from_exp)
    result["projects"] = extract_projects(_texts(sections, "projects"))
    result["certifications"] = extract_certifications(_texts(sections, "certifications"))
    result["achievements"] = extract_achievements(_texts(sections, "achievements"))

    sizes = {k: sum(1 for l in v if not l.is_blank) for k, v in sections.items()}
    out = validate_resume(result, ocr_confidence, settings.min_text_chars, settings.good_ocr_conf, sizes)
    if notes and out["extraction_status"] != "failed":
        out["warnings"].extend(notes)
        out["extraction_status"] = "needs_review"
    return out


def ocr_document(doc: LoadedResume, settings: Settings) -> tuple[List[Line], float, int]:
    """Stages 3-5 for every page. Returns (lines, mean confidence, columns_pages)."""
    all_lines: List[Line] = []
    confs, weights, col_pages = [], [], 0
    for page_no, img in enumerate(doc.images):
        proc = preprocess_image(img, settings)
        page: PageOCR = perform_ocr(proc, settings, page=page_no)
        if page_no > 0 and all_lines:
            all_lines.append(Line("", page_no))
        all_lines.extend(page.lines)
        if page.word_count:
            confs.append(page.mean_conf)
            weights.append(page.word_count)
        col_pages += int(page.columns_detected)
    mean = sum(c * w for c, w in zip(confs, weights)) / sum(weights) if weights else 0.0
    return all_lines, mean, col_pages


def process_resume(source: Union[str, os.PathLike, bytes], filename: Optional[str] = None,
                   settings: Optional[Settings] = None) -> Dict[str, Any]:
    """Main entry point. Never raises for bad input - returns status 'failed' with a warning.

    `source` may be a file path or raw bytes (e.g. from a FastAPI UploadFile).
    """
    settings = settings or Settings()
    result = empty_result()
    try:
        doc = load_resume(source, filename, settings)
        lines, conf, col_pages = ocr_document(doc, settings)
    except (ResumeLoadError, OCRError) as exc:
        result["extraction_status"] = "failed"
        result["warnings"] = [str(exc)]
        return result
    except Exception as exc:  # unexpected: keep the service alive, surface for review
        log.exception("Unexpected failure while processing resume")
        result["extraction_status"] = "failed"
        result["warnings"] = [f"Unexpected error: {type(exc).__name__}: {exc}"]
        return result

    result = extract_from_lines(lines, conf, settings)
    result["page_count"] = len(doc.images)
    if doc.truncated:
        result["warnings"].append(
            f"Document has {doc.total_pages} pages; only the first {len(doc.images)} were processed.")
        result["extraction_status"] = "needs_review"
    if col_pages:
        log.info("two-column layout detected on %d page(s)", col_pages)
    return result


def process_resume_bytes(data: bytes, filename: str, settings: Optional[Settings] = None) -> Dict[str, Any]:
    """FastAPI-friendly alias: process_resume_bytes(await upload.read(), upload.filename)."""
    return process_resume(data, filename, settings)


def process_resume_text(text: str, settings: Optional[Settings] = None) -> Dict[str, Any]:
    """Run extraction on plain text (tests, or PDFs that already have a text layer)."""
    return extract_from_lines(lines_from_text(text), None, settings)


# ==============================================================================
# COMMAND LINE
# ==============================================================================
if __name__ == "__main__":
    import argparse
    import json
    import sys

    ap = argparse.ArgumentParser(description="Extract structured data from a resume (PDF/JPG/PNG).")
    ap.add_argument("file")
    ap.add_argument("--no-raw", action="store_true", help="omit raw_text from the output")
    ap.add_argument("-o", "--output", help="write JSON to this file instead of stdout")
    args = ap.parse_args()
    out = process_resume(args.file)
    if args.no_raw:
        out.pop("raw_text", None)
    text = json.dumps(out, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text + "\n")
