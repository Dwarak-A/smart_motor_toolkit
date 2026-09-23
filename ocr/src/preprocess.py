import cv2
import numpy as np


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def find_plate_contour(gray):
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)

    if len(approx) == 4:
        return approx.reshape(4, 2)
    return None


def perspective_transform(image, pts):
    rect = order_points(pts.astype("float32"))
    tl, tr, br, bl = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))


def deskew(gray):
    coords = np.column_stack(np.where(gray < 255))
    if coords.size == 0:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    h, w = gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def apply_clahe(gray, clip_limit=3.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)


def binarize(gray, method="adaptive"):
    if method == "otsu":
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
        )
    return thresh


def is_low_contrast(gray, threshold=25.0):
    return gray.std() < threshold


def preprocess_image(image_path, attempt_perspective_correction=True):
    """
    Returns dict with:
        gray_enhanced: CLAHE-enhanced grayscale, deskewed (best input for OCR)
        binary: binarized version (fallback / secondary pass)
        low_contrast: bool flag, True if plate likely embossed/stamped
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    if attempt_perspective_correction:
        gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        pts = find_plate_contour(gray_full)
        if pts is not None:
            image = perspective_transform(image, pts)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    low_contrast = is_low_contrast(gray)

    gray = deskew(gray)
    enhanced = apply_clahe(gray)
    binary = binarize(enhanced)

    return {
        "gray_enhanced": enhanced,
        "binary": binary,
        "low_contrast": low_contrast,
    }


if __name__ == "__main__":
    import sys

    result = preprocess_image(sys.argv[1])
    cv2.imwrite("enhanced.png", result["gray_enhanced"])
    cv2.imwrite("binary.png", result["binary"])
    print("low_contrast:", result["low_contrast"])
