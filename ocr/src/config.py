from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SYNONYMS_PATH = DATA_DIR / "synonyms.json"
STANDARD_VALUES_PATH = DATA_DIR / "standard_values.json"

# OCR
OCR_LANG = "en"
FIELD_MATCH_CUTOFF = 0.75  # min fuzzy-match score to accept a label match
UNIT_FALLBACK_ENABLED = True

# Confidence gating
MIN_OCR_CONFIDENCE = 0.70  # below this, field goes to manual confirm
MIN_VALIDATION_TO_AUTOPUSH = True  # require validate() pass before auto-push

# Physical validation tolerances
POWER_EQUATION_TOLERANCE = 0.25  # +/- 25% on P vs sqrt3*V*I*eta*cosphi check
ASSUMED_EFFICIENCY = 0.85
ASSUMED_POWER_FACTOR = 0.85
RPM_SLIP_TOLERANCE = 0.08  # rated RPM must be within 8% below sync speed
STANDARD_FREQUENCIES = [50, 60]  # Hz

# Fallback trip thresholds if OCR/validation fails outright
FAILSAFE_UNBALANCE_TRIP_PCT = 5.0  # tighter than typical default (~10%)
FAILSAFE_TIME_DELAY_S = 1.0
