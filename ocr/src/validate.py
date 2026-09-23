import json
import math

import config

_standards = None


def _load_standards():
    global _standards
    if _standards is None:
        with open(config.STANDARD_VALUES_PATH, "r") as f:
            _standards = json.load(f)
    return _standards


def check_voltage(value):
    standards = _load_standards()
    closest = min(standards["standard_voltages"], key=lambda v: abs(v - value))
    pct_off = abs(closest - value) / closest
    return pct_off < 0.05, closest


def infer_poles_from_rpm(rpm, freq=50):
    standards = _load_standards()
    key = f"sync_speeds_{freq}hz"
    sync_speeds = standards[key]
    best_poles, best_diff = None, float("inf")
    for poles, sync in sync_speeds.items():
        diff = abs(sync - rpm) / sync
        if diff < best_diff:
            best_poles, best_diff = poles, diff
    return int(best_poles), best_diff


def check_rpm(rpm):
    standards = _load_standards()
    for freq in standards["standard_frequencies"]:
        poles, diff = infer_poles_from_rpm(rpm, freq)
        if diff <= config.RPM_SLIP_TOLERANCE:
            return True, {"frequency": freq, "poles": poles, "slip_fraction": diff}
    return False, None


def check_power_consistency(power_kw, voltage, current):
    """
    Cross-checks P ~ sqrt(3) * V * I * pf * eta for a three-phase motor.
    Returns (ok, expected_power_kw)
    """
    expected_w = (
        math.sqrt(3)
        * voltage
        * current
        * config.ASSUMED_POWER_FACTOR
        * config.ASSUMED_EFFICIENCY
    )
    expected_kw = expected_w / 1000.0
    if expected_kw <= 0:
        return False, expected_kw
    pct_off = abs(expected_kw - power_kw) / expected_kw
    return pct_off <= config.POWER_EQUATION_TOLERANCE, expected_kw


def validate_fields(fields):
    """
    fields: dict from field_matcher.assign_fields()
    Returns dict: {field_name: {"valid": bool, "detail": ...}}
    plus an "overall" key summarizing cross-field checks.
    """
    report = {}

    if "rated_voltage" in fields:
        ok, closest = check_voltage(fields["rated_voltage"]["value"])
        report["rated_voltage"] = {"valid": ok, "closest_standard": closest}

    if "rated_rpm" in fields:
        ok, detail = check_rpm(fields["rated_rpm"]["value"])
        report["rated_rpm"] = {"valid": ok, "detail": detail}

    if all(k in fields for k in ("rated_power", "rated_voltage", "rated_current")):
        power = fields["rated_power"]["value"]
        voltage = fields["rated_voltage"]["value"]
        current = fields["rated_current"]["value"]
        ok, expected = check_power_consistency(power, voltage, current)
        report["power_consistency"] = {"valid": ok, "expected_kw": round(expected, 2)}

    all_valid = all(v.get("valid", True) for v in report.values())
    report["overall"] = {"valid": all_valid, "fields_checked": len(report)}

    return report


if __name__ == "__main__":
    fields = {
        "rated_voltage": {"value": 415, "confidence": 0.9},
        "rated_current": {"value": 12.5, "confidence": 0.9},
        "rated_rpm": {"value": 1450, "confidence": 0.8},
        "rated_power": {"value": 7.5, "confidence": 0.85},
    }
    print(validate_fields(fields))
