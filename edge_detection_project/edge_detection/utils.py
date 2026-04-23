"""
utils.py — Shared helper functions used across the edge detection pipeline.

Responsibilities:
  - Loading images from disk (supports .jpg, .png, .bmp, etc.)
  - Converting images to grayscale
  - Applying Gaussian Blur (noise reduction before detection)
  - Saving processed images to disk
  - Encoding images as base64 strings for Flask response / HTML embedding
"""

import os
import base64
import cv2
import numpy as np


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from a file path using OpenCV.

    Args:
        image_path: Absolute or relative path to the image file.

    Returns:
        BGR image as a NumPy ndarray.

    Raises:
        FileNotFoundError: If the path does not exist or OpenCV cannot decode it.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(
            f"OpenCV could not decode the image at: {image_path}. "
            "Ensure the file is a valid image format (JPG, PNG, BMP, etc.)."
        )

    return img


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert a BGR image to grayscale.

    Edge detection algorithms operate on single-channel (grayscale) images,
    so this conversion is always applied before any detection step.

    Args:
        image: BGR image (H x W x 3) as a NumPy ndarray.

    Returns:
        Grayscale image (H x W) as a NumPy ndarray.
    """
    # If the image is already single-channel, return as-is
    if len(image.shape) == 2:
        return image

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_gaussian_blur(
    gray_image: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 0,
) -> np.ndarray:
    """
    Apply Gaussian Blur to suppress noise before edge detection.

    Noise in images creates false edges. A Gaussian kernel smooths the image
    by averaging pixel values weighted by a Gaussian distribution, preserving
    genuine edges while attenuating high-frequency noise.

    Args:
        gray_image: Single-channel grayscale image.
        kernel_size: Size of the Gaussian kernel (must be odd). Default: 5.
        sigma: Standard deviation of the Gaussian. If 0, it is computed
               automatically from kernel_size.

    Returns:
        Blurred grayscale image (same shape as input).
    """
    # Ensure kernel_size is positive and odd
    if kernel_size % 2 == 0:
        kernel_size += 1

    return cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), sigma)


def save_image(image: np.ndarray, output_path: str) -> str:
    """
    Save a processed image to disk.

    Args:
        image: NumPy ndarray image (grayscale or BGR).
        output_path: Full path including filename (e.g., 'static/out/sobel.png').

    Returns:
        The absolute path where the image was saved.

    Raises:
        IOError: If OpenCV cannot write to the specified path.
    """
    # Create parent directories if they don't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    success = cv2.imwrite(output_path, image)

    if not success:
        raise IOError(f"Failed to write image to: {output_path}")

    return os.path.abspath(output_path)


def encode_image_base64(image: np.ndarray, ext: str = ".png") -> str:
    """
    Encode a NumPy image array as a Base64 string for embedding in HTML.

    This avoids saving images to disk just to display them on a web page.
    The returned string can be used directly in an <img> src attribute:
        <img src="data:image/png;base64,<returned_string>">

    Args:
        image: NumPy ndarray (grayscale or BGR).
        ext: File extension that determines the encoding format. Default: '.png'.

    Returns:
        Base64-encoded string of the image.
    """
    # Encode image to in-memory buffer
    success, buffer = cv2.imencode(ext, image)

    if not success:
        raise ValueError("cv2.imencode failed — could not encode image to buffer.")

    # Convert buffer to bytes and then to base64
    encoded_bytes = base64.b64encode(buffer.tobytes())
    return encoded_bytes.decode("utf-8")
