import numpy as np
from paddleocr import PaddleOCR

_ocr_engine = None


def get_engine(lang="en"):
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
    return _ocr_engine


def run_ocr(image, lang="en"):
    """
    image: numpy array (grayscale or binary), e.g. output of preprocess_image()
    Returns list of dicts: {text, confidence, box}
    box is 4 (x, y) corner points in image coordinates.
    """
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)

    engine = get_engine(lang)
    raw = engine.ocr(image, cls=True)

    results = []
    if not raw or raw[0] is None:
        return results

    for line in raw[0]:
        box, (text, conf) = line
        results.append(
            {
                "text": text,
                "confidence": float(conf),
                "box": box,
            }
        )
    return results


def run_ocr_multi(preprocessed, lang="en"):
    """
    Runs OCR on both the enhanced grayscale and binary variants,
    keeps the higher-confidence read per overlapping text region.
    preprocessed: dict from preprocess_image()
    """
    results_gray = run_ocr(preprocessed["gray_enhanced"], lang=lang)
    results_bin = run_ocr(preprocessed["binary"], lang=lang)

    return merge_results(results_gray, results_bin)


def box_center(box):
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def merge_results(results_a, results_b, dist_threshold=20):
    """
    Deduplicates overlapping detections between two OCR passes,
    keeping whichever has higher confidence.
    """
    merged = list(results_a)
    used = [False] * len(results_a)

    for rb in results_b:
        cb = box_center(rb["box"])
        matched = False
        for i, ra in enumerate(merged):
            ca = box_center(ra["box"])
            dist = ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5
            if dist < dist_threshold:
                matched = True
                if rb["confidence"] > ra["confidence"]:
                    merged[i] = rb
                break
        if not matched:
            merged.append(rb)

    return merged


if __name__ == "__main__":
    import sys
    from preprocess import preprocess_image

    pre = preprocess_image(sys.argv[1])
    results = run_ocr_multi(pre)
    for r in results:
        print(f"{r['confidence']:.2f}  {r['text']}")
