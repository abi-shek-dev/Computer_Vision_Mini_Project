"""
prewitt.py — Prewitt Edge Detection (Manual Kernel Implementation)

Theory:
    The Prewitt operator is similar to Sobel but uses simpler, uniform weighting:

        Kx = [[-1, 0, 1],       Ky = [[-1, -1, -1],
              [-1, 0, 1],              [ 0,  0,  0],
              [-1, 0, 1]]              [ 1,  1,  1]]

    Unlike Sobel (which emphasizes the centre pixel by weighting it 2×),
    Prewitt applies equal weights across all three rows/columns.
    This makes Prewitt faster but slightly more noise-sensitive.

    Convolution is performed manually using cv2.filter2D so students can
    see exactly how kernel-based filtering works under the hood.
"""

import cv2
import numpy as np


# ── Prewitt kernels defined as module-level constants ──────────────────────────

# Detects vertical edges (gradient in the X direction)
PREWITT_KERNEL_X = np.array(
    [[-1, 0, 1],
     [-1, 0, 1],
     [-1, 0, 1]],
    dtype=np.float32,
)

# Detects horizontal edges (gradient in the Y direction)
PREWITT_KERNEL_Y = np.array(
    [[-1, -1, -1],
     [ 0,  0,  0],
     [ 1,  1,  1]],
    dtype=np.float32,
)


def apply_prewitt(blurred_gray: np.ndarray) -> dict:
    """
    Apply Prewitt edge detection using manually defined convolution kernels.

    OpenCV does not provide a built-in Prewitt function, so we use
    cv2.filter2D — a generic 2-D linear filter — with our custom kernels.

    Args:
        blurred_gray: Gaussian-blurred single-channel (grayscale) image.

    Returns:
        A dict with keys:
            'prewitt_x'  → horizontal gradient (absolute, uint8)
            'prewitt_y'  → vertical gradient (absolute, uint8)
            'combined'   → magnitude sqrt(Gx²+Gy²), normalized to uint8
    """
    # Convert to float32 for accurate arithmetic during convolution
    gray_f32 = blurred_gray.astype(np.float32)

    # --- Apply X kernel (cv2.filter2D performs 2-D cross-correlation) ---
    # ddepth=-1 keeps the same depth as source (float32 → float32)
    prewitt_x_raw = cv2.filter2D(gray_f32, ddepth=-1, kernel=PREWITT_KERNEL_X)

    # --- Apply Y kernel ---
    prewitt_y_raw = cv2.filter2D(gray_f32, ddepth=-1, kernel=PREWITT_KERNEL_Y)

    # Take absolute values and convert to uint8 for display
    prewitt_x = cv2.convertScaleAbs(prewitt_x_raw)
    prewitt_y = cv2.convertScaleAbs(prewitt_y_raw)

    # --- Combine into gradient magnitude ---
    magnitude = np.sqrt(prewitt_x_raw**2 + prewitt_y_raw**2)

    # Normalize to 0–255
    combined = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return {
        "prewitt_x": prewitt_x,
        "prewitt_y": prewitt_y,
        "combined": combined,
    }
