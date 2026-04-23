"""
app.py — Flask Application Entry Point for the Smart Edge Detection Tool

This module wires together all edge detection sub-modules and exposes four routes:

    GET  /              → Main upload / results page
    POST /process       → Receives an uploaded image + parameters, runs all
                          detectors, and returns JSON with base64 image data
    POST /save          → Saves a specific processed image to disk and returns
                          it as a downloadable file attachment
    GET  /webcam_feed   → Streams real-time webcam frames with Canny edge
                          detection applied (multipart/x-mixed-replace stream)

Usage:
    python app.py
    Then open http://127.0.0.1:5000 in your browser.
"""

import io
import os
import time
import uuid
import threading

import cv2
import numpy as np
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_file,
    Response,
)

from edge_detection import (
    apply_sobel,
    apply_prewitt,
    apply_canny,
    to_grayscale,
    apply_gaussian_blur,
    save_image,
    encode_image_base64,
)

# ── App Configuration ──────────────────────────────────────────────────────────

app = Flask(__name__)

# Maximum upload size: 16 MB
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

# Directory where user-saved images are stored
SAVE_DIR = os.path.join("static", "saved")
os.makedirs(SAVE_DIR, exist_ok=True)

# Allowed image extensions
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "tiff", "webp"}

# Global storage for in-memory processed images (keyed by session token)
# Kept simple (no Redis/DB) since this is a mini-project
_image_cache: dict[str, dict[str, np.ndarray]] = {}
_cache_lock = threading.Lock()


# ── Helper Functions ───────────────────────────────────────────────────────────


def allowed_file(filename: str) -> bool:
    """Return True if the file extension is in the allowed set."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def file_to_numpy(file_storage) -> np.ndarray:
    """
    Decode an uploaded Flask FileStorage object into a BGR NumPy array
    without saving to disk first.
    """
    file_bytes = np.frombuffer(file_storage.read(), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode uploaded file as an image.")
    return img


def run_all_detectors(
    bgr_image: np.ndarray,
    blur_kernel: int,
    canny_low: int,
    canny_high: int,
    sobel_ksize: int,
) -> dict:
    """
    Run the full edge detection pipeline on a BGR image.

    Steps:
      1. Convert to grayscale
      2. Apply Gaussian blur (noise reduction)
      3. Sobel X, Sobel Y, Sobel combined
      4. Prewitt X, Prewitt Y, Prewitt combined
      5. Canny

    Returns a flat dict mapping result_name → numpy image (uint8).
    """
    gray = to_grayscale(bgr_image)
    blurred = apply_gaussian_blur(gray, kernel_size=blur_kernel)

    sobel_results = apply_sobel(blurred, ksize=sobel_ksize)
    prewitt_results = apply_prewitt(blurred)
    canny_result = apply_canny(blurred, low_threshold=canny_low, high_threshold=canny_high)

    # Resize original to display next to results (keep BGR for colour)
    return {
        "original_gray": gray,
        "blurred": blurred,
        "sobel_x": sobel_results["sobel_x"],
        "sobel_y": sobel_results["sobel_y"],
        "sobel_combined": sobel_results["combined"],
        "prewitt_x": prewitt_results["prewitt_x"],
        "prewitt_y": prewitt_results["prewitt_y"],
        "prewitt_combined": prewitt_results["combined"],
        "canny": canny_result,
        # Keep original BGR for display
        "_bgr_original": bgr_image,
    }


# ── Routes ─────────────────────────────────────────────────────────────────────


@app.route("/")
def index():
    """Serve the main single-page UI."""
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    """
    Handle image upload + parameter form, run all edge detectors, and
    return a JSON payload with:
      - base64-encoded images for each detector output
      - a session token for subsequent /save requests
    """
    # ── Validate uploaded file ──
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {ALLOWED_EXTENSIONS}"}), 400

    # ── Parse parameters (with safe fallbacks) ──
    try:
        blur_kernel = int(request.form.get("blur_kernel", 5))
        canny_low = int(request.form.get("canny_low", 50))
        canny_high = int(request.form.get("canny_high", 150))
        sobel_ksize = int(request.form.get("sobel_ksize", 3))

        # Validate ranges
        blur_kernel = max(1, min(blur_kernel, 31))
        canny_low = max(0, min(canny_low, 255))
        canny_high = max(0, min(canny_high, 255))

        # Ensure Sobel ksize is valid (1, 3, 5, or 7)
        if sobel_ksize not in (1, 3, 5, 7):
            sobel_ksize = 3

        # Swap if thresholds are inverted
        if canny_low > canny_high:
            canny_low, canny_high = canny_high, canny_low

    except (ValueError, TypeError) as exc:
        return jsonify({"error": f"Invalid parameter: {exc}"}), 400

    # ── Decode image ──
    try:
        bgr = file_to_numpy(file)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    # ── Run detection pipeline ──
    results = run_all_detectors(bgr, blur_kernel, canny_low, canny_high, sobel_ksize)

    # ── Encode all outputs as base64 ──
    encoded = {}
    for key, img in results.items():
        if key.startswith("_"):
            continue  # skip internal-only entries
        encoded[key] = encode_image_base64(img)

    # Always include the colour original separately
    encoded["original_color"] = encode_image_base64(results["_bgr_original"])

    # ── Cache images for /save ──
    session_token = str(uuid.uuid4())
    with _cache_lock:
        # Evict old sessions if cache grows too large (simple LRU-lite)
        if len(_image_cache) > 20:
            oldest_key = next(iter(_image_cache))
            del _image_cache[oldest_key]
        _image_cache[session_token] = {k: v for k, v in results.items()}

    encoded["session_token"] = session_token
    encoded["image_shape"] = list(bgr.shape[:2])  # [height, width]

    return jsonify(encoded)


@app.route("/save", methods=["POST"])
def save():
    """
    Save a specific processed image and return it as a downloadable PNG.

    Expects JSON body:
      { "session_token": "<token>", "image_key": "<key>" }
    """
    data = request.get_json(force=True)
    session_token = data.get("session_token", "")
    image_key = data.get("image_key", "")

    with _cache_lock:
        session_data = _image_cache.get(session_token)

    if session_data is None:
        return jsonify({"error": "Session expired or invalid. Please re-upload your image."}), 404

    # Strip internal prefix if accidentally passed
    clean_key = image_key.lstrip("_")

    img = session_data.get(clean_key) or session_data.get(f"_{clean_key}")

    if img is None:
        available = [k for k in session_data if not k.startswith("_")]
        return jsonify({"error": f"Unknown image key '{image_key}'. Available: {available}"}), 400

    # ── Encode to PNG in memory and stream back ──
    success, buffer = cv2.imencode(".png", img)
    if not success:
        return jsonify({"error": "Failed to encode image."}), 500

    filename = f"edge_{clean_key}_{int(time.time())}.png"

    return send_file(
        io.BytesIO(buffer.tobytes()),
        mimetype="image/png",
        as_attachment=True,
        download_name=filename,
    )


# ── Webcam Streaming (Bonus Feature) ──────────────────────────────────────────

def _webcam_generator(canny_low: int = 50, canny_high: int = 150):
    """
    Generator that yields MJPEG frames from the default webcam with
    real-time Canny edge detection applied.

    Uses multipart/x-mixed-replace MIME type so the browser renders it
    as a continuous video stream inside a standard <img> tag.
    """
    cap = cv2.VideoCapture(0)  # 0 = default webcam

    if not cap.isOpened():
        # Yield a placeholder black frame if no webcam is found
        placeholder = np.zeros((480, 640), dtype=np.uint8)
        cv2.putText(
            placeholder,
            "No webcam found",
            (120, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            255,
            2,
        )
        _, buf = cv2.imencode(".jpg", placeholder)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Grayscale → blur → Canny
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, canny_low, canny_high)

            # Stack original (converted to BGR) and edges side by side
            edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            combined_frame = np.hstack([frame, edges_bgr])

            _, jpeg = cv2.imencode(".jpg", combined_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpeg.tobytes()
                + b"\r\n"
            )

            # ~30 FPS cap
            time.sleep(0.033)
    finally:
        cap.release()


@app.route("/webcam_feed")
def webcam_feed():
    """
    Streaming route for real-time webcam edge detection.
    Query params: low (int), high (int) — Canny thresholds.
    """
    low = int(request.args.get("low", 50))
    high = int(request.args.get("high", 150))
    low = max(0, min(low, 255))
    high = max(0, min(high, 255))
    if low > high:
        low, high = high, low

    return Response(
        _webcam_generator(low, high),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Smart Edge Detection Tool")
    print("  Running at: http://127.0.0.1:5000")
    print("=" * 60)
    # debug=False in production; set True for development auto-reload
    app.run(host="127.0.0.1", port=5000, debug=True)
