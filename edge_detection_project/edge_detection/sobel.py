"""
sobel.py — Sobel Edge Detection

Theory:
    The Sobel operator computes image gradients using two 3x3 convolution kernels:

        Kx = [[-1, 0, 1],       Ky = [[-1, -2, -1],
              [-2, 0, 2],              [ 0,  0,  0],
              [-1, 0, 1]]              [ 1,  2,  1]]

    Kx detects horizontal edges (vertical gradient).
    Ky detects vertical edges (horizontal gradient).

    The combined magnitude is: G = sqrt(Gx² + Gy²)

    Sobel uses a 3x3 Gaussian smoothing + differentiation combined kernel,
    so it is less sensitive to noise than simple finite-difference methods.
"""

import cv2
import numpy as np


def apply_sobel(
    blurred_gray: np.ndarray,
    ksize: int = 3,
    scale: float = 1.0,
    delta: float = 0.0,
) -> dict:
    """
    Apply Sobel edge detection and return X, Y, and combined magnitude maps.

    Args:
        blurred_gray: Gaussian-blurred single-channel (grayscale) image.
        ksize: Size of the Sobel kernel (1, 3, 5, or 7). Default: 3.
        scale: Optional scale factor applied to the computed derivative. Default: 1.
        delta: Optional delta added to the result before conversion. Default: 0.

    Returns:
        A dict with keys:
            'sobel_x'   → horizontal gradient (absolute, uint8)
            'sobel_y'   → vertical gradient (absolute, uint8)
            'combined'  → magnitude sqrt(Gx²+Gy²), normalized to uint8
    """
    # --- Sobel X (detects vertical edges — gradients in horizontal direction) ---
    # cv2.CV_64F ensures we capture both positive and negative gradient values
    sobel_x_raw = cv2.Sobel(
        blurred_gray,
        ddepth=cv2.CV_64F,
        dx=1,         # x-derivative
        dy=0,
        ksize=ksize,
        scale=scale,
        delta=delta,
    )

    # --- Sobel Y (detects horizontal edges — gradients in vertical direction) ---
    sobel_y_raw = cv2.Sobel(
        blurred_gray,
        ddepth=cv2.CV_64F,
        dx=0,
        dy=1,         # y-derivative
        ksize=ksize,
        scale=scale,
        delta=delta,
    )

    # Convert to absolute values and cast to uint8 for display
    sobel_x = cv2.convertScaleAbs(sobel_x_raw)
    sobel_y = cv2.convertScaleAbs(sobel_y_raw)

    # --- Combine X and Y gradients into full gradient magnitude ---
    # G = sqrt(Gx² + Gy²)  — computed in float64 to prevent overflow
    magnitude = np.sqrt(sobel_x_raw**2 + sobel_y_raw**2)

    # Normalize to 0–255 for display
    combined = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return {
        "sobel_x": sobel_x,
        "sobel_y": sobel_y,
        "combined": combined,
    }
