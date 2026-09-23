import config
import preprocess
import detect_recognize
import field_matcher
import validate


def run_pipeline(image_path):
    """
    Single entry point: image path -> result dict.

    Result shape:
    {
        "fields": {field: {"value", "confidence", "raw_text"}},
        "validation": {field: {"valid", ...}, "overall": {...}},
        "needs_manual_review": bool,
        "low_contrast_plate": bool,
        "failsafe": bool   # True if pipeline failed hard, caller should use
                            # config.FAILSAFE_* thresholds instead
    }
    """
    try:
        pre = preprocess.preprocess_image(image_path)
    except Exception as e:
        return _failsafe_result(reason=f"preprocess_failed: {e}")

    ocr_results = detect_recognize.run_ocr_multi(pre, lang=config.OCR_LANG)
    if not ocr_results:
        return _failsafe_result(reason="no_text_detected")

    fields = field_matcher.assign_fields(ocr_results)
    if not fields:
        return _failsafe_result(reason="no_fields_matched")

    validation = validate.validate_fields(fields)

    needs_review = _needs_manual_review(fields, validation)

    return {
        "fields": fields,
        "validation": validation,
        "needs_manual_review": needs_review,
        "low_contrast_plate": pre["low_contrast"],
        "failsafe": False,
    }


def _needs_manual_review(fields, validation):
    for name, f in fields.items():
        if f["confidence"] < config.MIN_OCR_CONFIDENCE:
            return True
    if config.MIN_VALIDATION_TO_AUTOPUSH:
        if not validation.get("overall", {}).get("valid", False):
            return True
    return False


def _failsafe_result(reason):
    return {
        "fields": {},
        "validation": {},
        "needs_manual_review": True,
        "low_contrast_plate": None,
        "failsafe": True,
        "failsafe_reason": reason,
        "failsafe_thresholds": {
            "unbalance_trip_pct": config.FAILSAFE_UNBALANCE_TRIP_PCT,
            "time_delay_s": config.FAILSAFE_TIME_DELAY_S,
        },
    }


if __name__ == "__main__":
    import sys
    import json

    result = run_pipeline(sys.argv[1])
    print(json.dumps(result, indent=2, default=str))
