# edge_detection package
# Exposes all detection functions for easy import in app.py

from .sobel import apply_sobel
from .prewitt import apply_prewitt
from .canny import apply_canny
from .utils import (
    load_image,
    to_grayscale,
    apply_gaussian_blur,
    save_image,
    encode_image_base64,
)
