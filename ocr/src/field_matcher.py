import json
import re
from difflib import SequenceMatcher

import config

_synonyms = None

UNIT_PATTERNS = {
    "rated_voltage": re.compile(r"\bV(?:AC)?\b", re.IGNORECASE),
    "rated_current": re.compile(r"\bA(?:MP)?S?\b", re.IGNORECASE),
    "rated_rpm": re.compile(r"\bR\.?P\.?M\.?\b|MIN-?1", re.IGNORECASE),
    "rated_power": re.compile(r"\bK?W\b|\bHP\b", re.IGNORECASE),
    "frequency": re.compile(r"\bHZ\b", re.IGNORECASE),
}

NUMBER_PATTERN = re.compile(r"[-+]?\d*\.?\d+")


def _load_synonyms():
    global _synonyms
    if _synonyms is None:
        with open(config.SYNONYMS_PATH, "r") as f:
            raw = json.load(f)
        # flatten to {synonym_upper: field_name}
        _synonyms = {}
        for field, syns in raw.items():
            for s in syns:
                _synonyms[s.upper()] = field
    return _synonyms


def normalize_token(text):
    return re.sub(r"[^A-Z0-9.\-]", "", text.upper())


def fuzzy_score(a, b):
    return SequenceMatcher(None, a, b).ratio()


def match_label(text, cutoff=None):
    """
    Attempts to match a raw OCR label string to a known field name.
    Returns (field_name, score) or (None, 0.0)
    """
    cutoff = cutoff if cutoff is not None else config.FIELD_MATCH_CUTOFF
    synonyms = _load_synonyms()
    token = normalize_token(text)

    if token in synonyms:
        return synonyms[token], 1.0

    best_field, best_score = None, 0.0
    for syn, field in synonyms.items():
        score = fuzzy_score(token, syn)
        if score > best_score:
            best_field, best_score = field, score

    if best_score >= cutoff:
        return best_field, best_score
    return None, 0.0


def match_by_unit(text):
    """
    Fallback: infer field type from a unit token adjacent to a number,
    used when the label itself is unreadable/garbled.
    """
    if not config.UNIT_FALLBACK_ENABLED:
        return None
    for field, pattern in UNIT_PATTERNS.items():
        if pattern.search(text):
            return field
    return None


def extract_number(text):
    match = NUMBER_PATTERN.search(text.replace(",", "."))
    return float(match.group()) if match else None


def assign_fields(ocr_results):
    """
    ocr_results: list of {text, confidence, box} from detect_recognize.run_ocr_multi
    Returns dict: {field_name: {"value": float|str, "confidence": float, "raw_text": str}}
    Simple strategy: for each detection, try label match on the token itself;
    if it also contains a number, assign directly. Otherwise try unit fallback.
    """
    fields = {}

    for r in ocr_results:
        text = r["text"]
        conf = r["confidence"]

        number = extract_number(text)
        field, score = match_label(text)

        if field is None:
            field = match_by_unit(text)
            score = 0.5 if field else 0.0  # lower confidence, no direct label match

        if field is None:
            continue

        if number is None:
            continue  # label with no associated value in this token; pipeline can pair by proximity

        combined_conf = conf * score if score > 0 else conf * 0.5

        existing = fields.get(field)
        if existing is None or combined_conf > existing["confidence"]:
            fields[field] = {
                "value": number,
                "confidence": combined_conf,
                "raw_text": text,
            }

    return fields


if __name__ == "__main__":
    sample = [
        {"text": "FLA: 12.5A", "confidence": 0.92, "box": []},
        {"text": "440V", "confidence": 0.88, "box": []},
        {"text": "1450 RPM", "confidence": 0.81, "box": []},
    ]
    print(assign_fields(sample))
