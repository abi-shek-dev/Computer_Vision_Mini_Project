"""
canny.py — Canny Edge Detection

Theory:
    Canny is a multi-stage algorithm developed by John Canny (1986):

    1. Gaussian Smoothing  — Noise reduction (already applied in preprocessing).
    2. Gradient Computation — Compute intensity gradients (Sobel internally).
    3. Non-Maximum Suppression — Thin edges to 1-pixel wide ridges.
    4. Double Thresholding  — Classify pixels as strong, weak, or non-edge:
         • pixel ≥ high_threshold  → strong edge (definitely an edge)
         • low_threshold ≤ pixel < high_threshold → weak edge (maybe an edge)
         • pixel < low_threshold   → suppressed (not an edge)
    5. Edge Tracking by Hysteresis — Retain weak edges only if they connect to
       a strong edge, eliminating isolated noise responses.

    Result: Clean, thin, well-connected edges with minimal noise.
"""

import cv2
import numpy as np


def apply_canny(
    blurred_gray: np.ndarray,
    low_threshold: int = 50,
    high_threshold: int = 150,
    aperture_size: int = 3,
    l2_gradient: bool = False,
) -> np.ndarray:
    """
    Apply the Canny edge detection algorithm.

    Args:
        blurred_gray: Gaussian-blurred single-channel (grayscale) image.
                      Passing a pre-blurred image is recommended — OpenCV's
                      Canny can optionally blur internally but offering control
                      over the blur step separately gives better results.
        low_threshold:  Lower bound for double thresholding (hysteresis).
                        Pixels below this value are rejected as non-edges.
                        Typical range: 10–100.
        high_threshold: Upper bound for double thresholding.
                        Pixels above this value are accepted as strong edges.
                        Typical range: 100–300. Rule of thumb: ~3× low_threshold.
        aperture_size:  Size of the Sobel kernel used internally (3, 5, or 7).
                        Default: 3.
        l2_gradient:    If True, use the L2 norm (sqrt(Gx²+Gy²)) for gradient
                        magnitude; if False, use the faster L1 norm (|Gx|+|Gy|).
                        Default: False.

    Returns:
        Binary edge map (uint8) where 255 = edge pixel, 0 = background.
    """
    # cv2.Canny expects a uint8 image
    if blurred_gray.dtype != np.uint8:
        blurred_gray = blurred_gray.astype(np.uint8)

    edges = cv2.Canny(
        blurred_gray,
        threshold1=low_threshold,
        threshold2=high_threshold,
        apertureSize=aperture_size,
        L2gradient=l2_gradient,
    )

    return edges


def auto_thresholds(gray_image: np.ndarray, sigma: float = 0.33) -> tuple[int, int]:
    """
    Automatically estimate Canny thresholds based on the median pixel intensity.

    This heuristic (by Adrian Rosebrock) computes a reasonable threshold pair
    that works well across a wide range of images without manual tuning:
        v          = median pixel intensity
        low        = max(0,   (1 - sigma) * v)
        high       = min(255, (1 + sigma) * v)

    Args:
        gray_image: Single-channel grayscale image.
        sigma: Controls how wide the threshold window is. Default: 0.33.

    Returns:
        Tuple of (low_threshold, high_threshold) as integers.
    """
    v = float(np.median(gray_image))
    low  = int(max(0,   (1.0 - sigma) * v))
    high = int(min(255, (1.0 + sigma) * v))
    return low, high
