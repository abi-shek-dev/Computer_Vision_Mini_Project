# Smart Edge Detection Tool for Image Analysis

A complete Python + Flask web application that lets you upload an image and compare **Sobel**, **Prewitt**, and **Canny** edge detection algorithms side-by-side. Includes real-time webcam edge detection and one-click PNG / ZIP download of results.

---

## Features

| Feature | Detail |
|---|---|
| **Sobel Detection** | X, Y, and combined magnitude with configurable kernel size (1/3/5/7) |
| **Prewitt Detection** | Manual kernel implementation using `cv2.filter2D` |
| **Canny Detection** | Full hysteresis with adjustable low/high thresholds |
| **Gaussian Blur** | Configurable pre-processing noise reduction |
| **Side-by-side comparison** | Tabbed results view: Overview, Sobel, Prewitt, Canny |
| **Lightbox viewer** | Click any result to view full-size |
| **Individual download** | Save any output as PNG |
| **Download All** | Export all 9 outputs as a ZIP archive |
| **Real-time webcam** | Live Canny edge detection on webcam feed (MJPEG stream) |

---

## Project Structure

```
edge_detection_project/
├── app.py                    # Flask application & all routes
├── requirements.txt          # Python dependencies
├── README.md
├── edge_detection/
│   ├── __init__.py           # Package exports
│   ├── utils.py              # load, grayscale, blur, save, base64 helpers
│   ├── sobel.py              # Sobel X / Y / combined
│   ├── prewitt.py            # Prewitt X / Y / combined (manual kernels)
│   └── canny.py              # Canny + auto-threshold helper
├── templates/
│   └── index.html            # Jinja2 single-page UI
└── static/
    ├── css/
    │   └── style.css
    └── js/
        └── app.js
```

---

## Installation

### 1. Clone / navigate to the project

```bash
cd edge_detection_project
```

### 2. Create and activate a virtual environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## How to Run

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

---

## API Routes

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Main UI page |
| `POST` | `/process` | Upload image + params → JSON of base64 results |
| `POST` | `/save` | Download one processed image as PNG |
| `GET` | `/webcam_feed` | MJPEG stream of real-time Canny edge detection |

### `/process` parameters (multipart/form-data)

| Field | Type | Default | Description |
|---|---|---|---|
| `image` | File | — | Image file (JPG/PNG/BMP/TIFF/WEBP, max 16 MB) |
| `blur_kernel` | int | 5 | Gaussian blur kernel size (odd, 1–31) |
| `canny_low` | int | 50 | Canny lower hysteresis threshold (0–255) |
| `canny_high` | int | 150 | Canny upper hysteresis threshold (0–255) |
| `sobel_ksize` | int | 3 | Sobel kernel size (1, 3, 5, or 7) |

---

## Algorithm Reference

### Sobel

Computes image gradients using:

```
Kx = [[-1, 0, +1],    Ky = [[-1, -2, -1],
      [-2, 0, +2],          [ 0,  0,  0],
      [-1, 0, +1]]          [+1, +2, +1]]
```

Combined magnitude: `G = √(Gx² + Gy²)`

### Prewitt

Same concept but with uniform (non-weighted) kernels — implemented manually via `cv2.filter2D`:

```
Kx = [[-1, 0, +1],    Ky = [[-1, -1, -1],
      [-1, 0, +1],          [ 0,  0,  0],
      [-1, 0, +1]]          [+1, +1, +1]]
```

### Canny

Multi-stage algorithm:
1. Gaussian smoothing (pre-applied)
2. Gradient computation (internal Sobel)
3. Non-maximum suppression (thin edges to 1-px wide)
4. Double thresholding (strong / weak / non-edge)
5. Hysteresis edge tracking

---

## Requirements

```
flask>=3.0.0
opencv-python>=4.9.0
numpy>=1.26.0
```

Python **3.11+** recommended.

---

## Screenshots

| | |
|---|---|
| Upload & configure | Side-by-side comparison |
| Sobel X / Y detail | Real-time webcam feed |

---

## License

MIT — free for academic and personal use.
