/**
 * app.js — Front-end logic for the Smart Edge Detection Tool
 *
 * Responsibilities:
 *   - Drag-and-drop & file-picker image upload
 *   - Slider / radio-button parameter controls (live value display)
 *   - POST to /process and populate result images from base64 response
 *   - Tab switching in the results section
 *   - Lightbox for full-screen image viewing
 *   - Per-image PNG download via POST /save
 *   - "Download All" as a ZIP (using JSZip from CDN)
 *   - Real-time webcam streaming start / stop
 *   - Animated hero pixel grid
 *   - Toast notification helper
 */

"use strict";

/* ═══════════════════════════════════════════════════════════
   HERO PIXEL GRID
═══════════════════════════════════════════════════════════ */
(function initPixelGrid() {
  const grid = document.getElementById("heroPixelGrid");
  if (!grid) return;

  const COLORS = ["#3b82f6", "#a855f7", "#06b6d4", "#10b981", "#f59e0b"];
  const COUNT = 14 * 14; // 196 cells

  for (let i = 0; i < COUNT; i++) {
    const cell = document.createElement("div");
    cell.className = "pixel-cell";

    const color = COLORS[Math.floor(Math.random() * COLORS.length)];
    const delay = (Math.random() * 3).toFixed(2);
    const dur   = (2 + Math.random() * 2).toFixed(2);

    cell.style.cssText = `
      background: ${color};
      animation-delay: ${delay}s;
      animation-duration: ${dur}s;
    `;

    grid.appendChild(cell);
  }
})();


/* ═══════════════════════════════════════════════════════════
   TOAST HELPER
═══════════════════════════════════════════════════════════ */
function showToast(message, type = "info") {
  const container = document.getElementById("toastContainer");
  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.textContent = message;
  container.appendChild(toast);

  // Trigger animation on next frame
  requestAnimationFrame(() => {
    requestAnimationFrame(() => toast.classList.add("show"));
  });

  setTimeout(() => {
    toast.classList.remove("show");
    setTimeout(() => toast.remove(), 350);
  }, 3500);
}


/* ═══════════════════════════════════════════════════════════
   SLIDER CONTROLS — live value display
═══════════════════════════════════════════════════════════ */
function bindSlider(sliderId, displayId) {
  const slider  = document.getElementById(sliderId);
  const display = document.getElementById(displayId);
  if (!slider || !display) return;

  const update = () => { display.textContent = slider.value; };
  slider.addEventListener("input", update);
  update();
}

bindSlider("blurKernel",  "blurKernelVal");
bindSlider("cannyLow",    "cannyLowVal");
bindSlider("cannyHigh",   "cannyHighVal");
bindSlider("webcamLow",   "webcamLowVal");
bindSlider("webcamHigh",  "webcamHighVal");


/* ═══════════════════════════════════════════════════════════
   SOBEL KERNEL SIZE — radio button group
═══════════════════════════════════════════════════════════ */
let sobelKsize = 3;

(function initSobelButtons() {
  const group = document.getElementById("sobelKsizeGroup");
  if (!group) return;

  // Set default selection to "3"
  const defaultBtn = group.querySelector('[data-value="3"]');
  if (defaultBtn) {
    group.querySelectorAll(".radio-btn").forEach(b => b.classList.remove("selected"));
    defaultBtn.classList.add("selected");
  }

  group.addEventListener("click", (e) => {
    const btn = e.target.closest(".radio-btn");
    if (!btn) return;
    group.querySelectorAll(".radio-btn").forEach(b => b.classList.remove("selected"));
    btn.classList.add("selected");
    sobelKsize = parseInt(btn.dataset.value, 10);
    document.getElementById("sobelKsizeVal").textContent = sobelKsize;
  });
})();


/* ═══════════════════════════════════════════════════════════
   IMAGE UPLOAD — drag-and-drop + file picker
═══════════════════════════════════════════════════════════ */
const dropZone      = document.getElementById("dropZone");
const imageInput    = document.getElementById("imageInput");
const previewWrapper = document.getElementById("previewWrapper");
const previewImg    = document.getElementById("previewImg");
const previewMeta   = document.getElementById("previewMeta");
const clearBtn      = document.getElementById("clearBtn");
const processBtn    = document.getElementById("processBtn");

let uploadedFile = null; // currently selected File object

function handleFile(file) {
  if (!file) return;

  // Validate type client-side
  const allowed = ["image/jpeg", "image/png", "image/bmp", "image/tiff", "image/webp"];
  if (!allowed.includes(file.type) && !file.name.match(/\.(jpg|jpeg|png|bmp|tiff|tif|webp)$/i)) {
    showToast("Unsupported file type. Please upload JPG, PNG, BMP, TIFF, or WEBP.", "error");
    return;
  }

  uploadedFile = file;

  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    dropZone.style.display = "none";
    previewWrapper.style.display = "flex";

    previewMeta.textContent =
      `${file.name}  |  ${(file.size / 1024).toFixed(1)} KB  |  ${file.type}`;

    processBtn.disabled = false;
    showToast("Image loaded! Configure parameters and click Run.", "info");
  };
  reader.readAsDataURL(file);
}

// Click to open file picker
dropZone.addEventListener("click", () => imageInput.click());
imageInput.addEventListener("change", () => handleFile(imageInput.files[0]));

// Drag-and-drop
dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  handleFile(e.dataTransfer.files[0]);
});

// Clear button
clearBtn.addEventListener("click", () => {
  uploadedFile = null;
  imageInput.value = "";
  previewImg.src = "";
  previewWrapper.style.display = "none";
  dropZone.style.display = "flex";
  processBtn.disabled = true;
});


/* ═══════════════════════════════════════════════════════════
   PROCESS — send image + params to /process
═══════════════════════════════════════════════════════════ */
// Stores the session token returned by the server for subsequent /save calls
let currentSessionToken = null;
// Full b64 map stored locally to enable "download all" without another round-trip
let currentResults = {};

processBtn.addEventListener("click", async () => {
  if (!uploadedFile) {
    showToast("Please upload an image first.", "error");
    return;
  }

  const blurKernel  = document.getElementById("blurKernel").value;
  const cannyLow    = document.getElementById("cannyLow").value;
  const cannyHigh   = document.getElementById("cannyHigh").value;

  const formData = new FormData();
  formData.append("image",       uploadedFile);
  formData.append("blur_kernel", blurKernel);
  formData.append("canny_low",   cannyLow);
  formData.append("canny_high",  cannyHigh);
  formData.append("sobel_ksize", sobelKsize);

  // Show loading overlay
  document.getElementById("loadingOverlay").style.display = "flex";

  try {
    const response = await fetch("/process", { method: "POST", body: formData });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Server error.");
    }

    // Store session info
    currentSessionToken = data.session_token;
    currentResults = data;

    // Populate all result images
    populateResults(data);

    // Show results section and scroll to it
    const resultsSection = document.getElementById("results-section");
    resultsSection.style.display = "block";
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });

    showToast("Processing complete!", "success");

  } catch (err) {
    console.error("Process error:", err);
    showToast(`Error: ${err.message}`, "error");
  } finally {
    document.getElementById("loadingOverlay").style.display = "none";
  }
});


/* ── Populate Result Images ─────────────────────────────── */
function setResultImage(imgId, b64Data, mimeType = "image/png") {
  const el = document.getElementById(imgId);
  if (!el) return;
  el.src = `data:${mimeType};base64,${b64Data}`;
}

function populateResults(data) {
  // Overview tab
  setResultImage("img-original_color",     data.original_color);
  setResultImage("img-sobel_combined",      data.sobel_combined);
  setResultImage("img-prewitt_combined",    data.prewitt_combined);
  setResultImage("img-canny",               data.canny);

  // Sobel tab
  setResultImage("img-sobel_x",            data.sobel_x);
  setResultImage("img-sobel_y",            data.sobel_y);
  setResultImage("img-sobel_combined_tab", data.sobel_combined);

  // Prewitt tab
  setResultImage("img-prewitt_x",            data.prewitt_x);
  setResultImage("img-prewitt_y",            data.prewitt_y);
  setResultImage("img-prewitt_combined_tab", data.prewitt_combined);

  // Canny tab
  setResultImage("img-original_gray_canny", data.original_gray);
  setResultImage("img-canny_tab",            data.canny);

  // Canny params info panel
  const low  = document.getElementById("cannyLow").value;
  const high = document.getElementById("cannyHigh").value;
  const [h, w] = data.image_shape;
  document.getElementById("cannyParamsDisplay").textContent =
    `Low threshold: ${low}  |  High threshold: ${high}  |  Image: ${w} × ${h} px  |  Blur kernel: ${document.getElementById("blurKernel").value}`;

  // Results meta bar
  document.getElementById("resultsMeta").textContent =
    `${w} × ${h} px — Sobel k=${sobelKsize}, Canny [${low}, ${high}], Blur k=${document.getElementById("blurKernel").value}`;
}


/* ═══════════════════════════════════════════════════════════
   TAB SWITCHING
═══════════════════════════════════════════════════════════ */
document.getElementById("tabBar").addEventListener("click", (e) => {
  const tab = e.target.closest(".tab");
  if (!tab) return;

  document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
  document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));

  tab.classList.add("active");
  const panelId = `tab-${tab.dataset.tab}`;
  document.getElementById(panelId).classList.add("active");
});


/* ═══════════════════════════════════════════════════════════
   LIGHTBOX — click any result card to enlarge
═══════════════════════════════════════════════════════════ */
const lightbox      = document.getElementById("lightbox");
const lightboxImg   = document.getElementById("lightboxImg");
const lightboxLabel = document.getElementById("lightboxLabel");
const lightboxClose = document.getElementById("lightboxClose");

document.addEventListener("click", (e) => {
  const card = e.target.closest(".result-card");
  if (!card) return;

  // Don't open lightbox when clicking the download button
  if (e.target.closest(".btn-icon-only")) return;

  const img = card.querySelector(".result-img");
  if (!img || !img.src) return;

  lightboxImg.src = img.src;
  lightboxLabel.textContent = card.querySelector(".result-label")?.textContent || "";
  lightbox.style.display = "flex";
  document.body.style.overflow = "hidden";
});

function closeLightbox() {
  lightbox.style.display = "none";
  document.body.style.overflow = "";
}
lightboxClose.addEventListener("click", closeLightbox);
lightbox.addEventListener("click", (e) => {
  if (e.target === lightbox) closeLightbox();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeLightbox();
});


/* ═══════════════════════════════════════════════════════════
   DOWNLOAD — individual image via /save
═══════════════════════════════════════════════════════════ */

// Map HTML image key → server-side key
const KEY_MAP = {
  original_color:     "original_color",  // encoded as BGR original
  sobel_combined:     "sobel_combined",
  prewitt_combined:   "prewitt_combined",
  canny:              "canny",
  sobel_x:            "sobel_x",
  sobel_y:            "sobel_y",
  prewitt_x:          "prewitt_x",
  prewitt_y:          "prewitt_y",
  original_gray:      "original_gray",
};

document.addEventListener("click", async (e) => {
  const btn = e.target.closest("[data-download]");
  if (!btn) return;

  if (!currentSessionToken) {
    showToast("Process an image first.", "error");
    return;
  }

  const imageKey = btn.dataset.download;

  try {
    const response = await fetch("/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_token: currentSessionToken,
        image_key: KEY_MAP[imageKey] || imageKey,
      }),
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.error || "Download failed.");
    }

    // Trigger browser download
    const blob = await response.blob();
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href = url;
    a.download = `edge_${imageKey}.png`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);

    showToast(`Downloaded: edge_${imageKey}.png`, "success");

  } catch (err) {
    console.error("Download error:", err);
    showToast(`Download error: ${err.message}`, "error");
  }
});


/* ═══════════════════════════════════════════════════════════
   DOWNLOAD ALL — client-side ZIP using JSZip
   JSZip is loaded from CDN below the script tag
═══════════════════════════════════════════════════════════ */
document.getElementById("downloadAllBtn").addEventListener("click", async () => {
  if (!currentSessionToken || Object.keys(currentResults).length === 0) {
    showToast("Process an image first.", "error");
    return;
  }

  showToast("Preparing ZIP…", "info");

  // Load JSZip dynamically (CDN)
  if (typeof JSZip === "undefined") {
    await new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = "https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js";
      s.onload = resolve;
      s.onerror = reject;
      document.head.appendChild(s);
    });
  }

  try {
    const zip = new JSZip();

    const keys = [
      "original_color", "original_gray", "blurred",
      "sobel_x", "sobel_y", "sobel_combined",
      "prewitt_x", "prewitt_y", "prewitt_combined",
      "canny",
    ];

    for (const key of keys) {
      const b64 = currentResults[key];
      if (!b64) continue;

      // Convert base64 → binary
      const binary = atob(b64);
      const bytes  = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);

      zip.file(`${key}.png`, bytes);
    }

    const blob = await zip.generateAsync({ type: "blob" });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href = url;
    a.download = "edge_detection_results.zip";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);

    showToast("All images downloaded as ZIP!", "success");
  } catch (err) {
    console.error("ZIP error:", err);
    showToast(`ZIP error: ${err.message}`, "error");
  }
});


/* ═══════════════════════════════════════════════════════════
   WEBCAM STREAMING
═══════════════════════════════════════════════════════════ */
const startWebcamBtn  = document.getElementById("startWebcamBtn");
const stopWebcamBtn   = document.getElementById("stopWebcamBtn");
const webcamWrapper   = document.getElementById("webcamWrapper");
const webcamFeed      = document.getElementById("webcamFeed");

let webcamActive = false;

startWebcamBtn.addEventListener("click", () => {
  const low  = document.getElementById("webcamLow").value;
  const high = document.getElementById("webcamHigh").value;

  webcamFeed.src = `/webcam_feed?low=${low}&high=${high}&t=${Date.now()}`;
  webcamWrapper.style.display = "block";
  startWebcamBtn.style.display = "none";
  stopWebcamBtn.style.display  = "";
  webcamActive = true;

  showToast("Webcam stream started.", "info");
});

stopWebcamBtn.addEventListener("click", () => {
  // Clearing src stops the MJPEG stream from being requested
  webcamFeed.src = "";
  webcamWrapper.style.display = "none";
  stopWebcamBtn.style.display  = "none";
  startWebcamBtn.style.display = "";
  webcamActive = false;

  showToast("Webcam stopped.", "info");
});

// Live threshold update for webcam (re-request stream with new params)
["webcamLow", "webcamHigh"].forEach((id) => {
  document.getElementById(id).addEventListener("change", () => {
    if (!webcamActive) return;
    const low  = document.getElementById("webcamLow").value;
    const high = document.getElementById("webcamHigh").value;
    webcamFeed.src = `/webcam_feed?low=${low}&high=${high}&t=${Date.now()}`;
  });
});
