/**
 * Ethiopian Birr AI Detector - In-Browser YOLOv8 Web Engine
 * 100% Client-Side WebAssembly / WebGL Inference
 */

// Denomination Configuration
const DENOMS = {
  0: { id: 0, name: '5', label: '5 Birr', val: 5, color: '#d97706', rgb: [217, 119, 6] },
  1: { id: 1, name: '10', label: '10 Birr', val: 10, color: '#10b981', rgb: [16, 185, 129] },
  2: { id: 2, name: '50', label: '50 Birr', val: 50, color: '#ef4444', rgb: [239, 68, 68] },
  3: { id: 3, name: '100', label: '100 Birr', val: 100, color: '#06b6d4', rgb: [6, 182, 212] },
  4: { id: 4, name: '200', label: '200 Birr', val: 200, color: '#f59e0b', rgb: [245, 158, 11] }
};

// Global App State
let ortSession = null;
let activeMode = 'upload'; // 'upload' or 'camera'
let cameraStream = null;
let cameraFacingMode = 'environment'; // 'environment' (back) or 'user' (front)
let isCameraRunning = false;
let cameraAnimationId = null;
let isProcessingFrame = false;
let currentImage = null; // Stored uploaded image
let lastSpokenTotal = -1;
let frameCount = 0;
let lastFpsTime = performance.now();

// DOM Elements
const statusPill = document.getElementById('statusPill');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const loadingOverlay = document.getElementById('loadingOverlay');
const loadingMsg = document.getElementById('loadingMsg');
const manualModelUpload = document.getElementById('manualModelUpload');
const modelFileInput = document.getElementById('modelFileInput');

const tabUploadBtn = document.getElementById('tabUploadBtn');
const tabCameraBtn = document.getElementById('tabCameraBtn');
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const chooseImageBtn = document.getElementById('chooseImageBtn');
const startCamBtn = document.getElementById('startCamBtn');
const switchCamBtn = document.getElementById('switchCamBtn');
const speakBtn = document.getElementById('speakBtn');
const downloadBtn = document.getElementById('downloadBtn');

const cameraVideo = document.getElementById('cameraVideo');
const displayCanvas = document.getElementById('displayCanvas');
const displayCtx = displayCanvas.getContext('2d');

const totalAmount = document.getElementById('totalAmount');
const totalNotesPill = document.getElementById('totalNotesPill');
const inferTime = document.getElementById('inferTime');
const fpsVal = document.getElementById('fpsVal');
const confSlider = document.getElementById('confSlider');
const confVal = document.getElementById('confVal');
const iouSlider = document.getElementById('iouSlider');
const iouVal = document.getElementById('iouVal');
const skinToggle = document.getElementById('skinToggle');
const autoVoiceToggle = document.getElementById('autoVoiceToggle');

// 1. Offscreen Canvas for YOLO 640x640 Input Normalization
const offscreenCanvas = document.createElement('canvas');
offscreenCanvas.width = 640;
offscreenCanvas.height = 640;
const offscreenCtx = offscreenCanvas.getContext('2d', { willReadFrequently: true });

// --- MODEL INITIALIZATION ---
async function initModel() {
  try {
    loadingMsg.innerText = 'Downloading In-Browser YOLOv8 Model (12.7 MB)...';
    
    // Configure ONNX Runtime Web
    ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 2);
    ort.env.wasm.simd = true;

    // Load from local model directory
    ortSession = await ort.InferenceSession.create('./model/best.onnx', {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all'
    });

    onModelReady();
  } catch (err) {
    console.warn('Auto-fetch of best.onnx failed (often due to file:// protocol restriction):', err);
    loadingMsg.innerText = 'Could not auto-load ./model/best.onnx';
    statusText.innerText = 'Manual Model Load Required';
    manualModelUpload.style.display = 'block';
  }
}

function onModelReady() {
  loadingOverlay.style.display = 'none';
  statusDot.classList.add('ready');
  statusText.innerText = 'YOLOv8 Model Ready';
  console.log('✅ YOLOv8 Model Loaded Successfully into Browser!');
}

// Fallback: If user opens file directly through file:// and browser blocks fetch
if (modelFileInput) {
  modelFileInput.addEventListener('change', async (e) => {
    if (e.target.files.length > 0) {
      try {
        loadingMsg.innerText = 'Loading uploaded model weights...';
        const file = e.target.files[0];
        const arrayBuffer = await file.arrayBuffer();
        ortSession = await ort.InferenceSession.create(arrayBuffer, {
          executionProviders: ['wasm']
        });
        onModelReady();
      } catch (uploadErr) {
        alert('Failed to parse ONNX file: ' + uploadErr.message);
      }
    }
  });
}

// --- YOLO PREPROCESSING ---
function preprocess(source, srcW, srcH) {
  // Draw scaled to 640x640
  offscreenCtx.drawImage(source, 0, 0, 640, 640);
  const imgData = offscreenCtx.getImageData(0, 0, 640, 640).data;

  // Float32 array in [1, 3, 640, 640] CHW format
  const area = 640 * 640;
  const floatData = new Float32Array(3 * area);

  for (let i = 0; i < area; i++) {
    const r = imgData[i * 4] / 255.0;
    const g = imgData[i * 4 + 1] / 255.0;
    const b = imgData[i * 4 + 2] / 255.0;

    floatData[i] = r;              // Channel 0 (Red)
    floatData[area + i] = g;       // Channel 1 (Green)
    floatData[2 * area + i] = b;   // Channel 2 (Blue)
  }

  return new ort.Tensor('float32', floatData, [1, 3, 640, 640]);
}

// --- BIOMETRIC YCrCb SKIN FILTER (Pure JS) ---
function isSkinRegion(sourceCtx, x1, y1, x2, y2) {
  const w = Math.max(1, Math.floor(x2 - x1));
  const h = Math.max(1, Math.floor(y2 - y1));
  if (w < 20 || h < 20) return false;

  // Sample bounding box pixels
  const imgData = sourceCtx.getImageData(Math.floor(x1), Math.floor(y1), w, h).data;
  let skinCount = 0;
  const totalPixels = w * h;

  // Sample every 4th pixel for high speed
  const step = 4;
  let sampled = 0;

  for (let i = 0; i < totalPixels * 4; i += 4 * step) {
    const r = imgData[i];
    const g = imgData[i + 1];
    const b = imgData[i + 2];

    // Standard YCrCb conversion
    const y = 0.299 * r + 0.587 * g + 0.114 * b;
    const cr = (r - y) * 0.713 + 128;
    const cb = (b - y) * 0.564 + 128;

    if (cr >= 135 && cr <= 175 && cb >= 80 && cb <= 125) {
      skinCount++;
    }
    sampled++;
  }

  return (skinCount / sampled) > 0.65;
}

// --- NON-MAXIMUM SUPPRESSION (NMS) ---
function computeIoU(b1, b2) {
  const x1 = Math.max(b1.x1, b2.x1);
  const y1 = Math.max(b1.y1, b2.y1);
  const x2 = Math.min(b1.x2, b2.x2);
  const y2 = Math.min(b1.y2, b2.y2);

  const interW = Math.max(0, x2 - x1);
  const interH = Math.max(0, y2 - y1);
  const interArea = interW * interH;

  const area1 = (b1.x2 - b1.x1) * (b1.y2 - b1.y1);
  const area2 = (b2.x2 - b2.x1) * (b2.y2 - b2.y1);
  const unionArea = area1 + area2 - interArea;

  return unionArea <= 0 ? 0 : interArea / unionArea;
}

function runNMS(boxes, iouThreshold) {
  boxes.sort((a, b) => b.score - a.score);
  const selected = [];

  for (const box of boxes) {
    let keep = true;
    for (const chosen of selected) {
      if (box.classId === chosen.classId) {
        if (computeIoU(box, chosen) > iouThreshold) {
          keep = false;
          break;
        }
      }
    }
    if (keep) {
      selected.push(box);
    }
  }
  return selected;
}

// --- YOLO POST-PROCESSING ---
function postprocess(outputData, srcW, srcH, confThreshold, iouThreshold) {
  // Output shape is [1, 9, 8400] -> 9 rows, 8400 columns
  const numCandidates = 8400;
  const candidates = [];

  const scaleX = srcW / 640.0;
  const scaleY = srcH / 640.0;

  for (let i = 0; i < numCandidates; i++) {
    // Class scores start at row 4 (indices: 4, 5, 6, 7, 8)
    let maxScore = -1;
    let bestClass = -1;

    for (let c = 0; c < 5; c++) {
      const score = outputData[(4 + c) * numCandidates + i];
      if (score > maxScore) {
        maxScore = score;
        bestClass = c;
      }
    }

    if (maxScore >= confThreshold) {
      const cx = outputData[0 * numCandidates + i];
      const cy = outputData[1 * numCandidates + i];
      const w  = outputData[2 * numCandidates + i];
      const h  = outputData[3 * numCandidates + i];

      const x1 = Math.max(0, (cx - w / 2) * scaleX);
      const y1 = Math.max(0, (cy - h / 2) * scaleY);
      const x2 = Math.min(srcW, (cx + w / 2) * scaleX);
      const y2 = Math.min(srcH, (cy + h / 2) * scaleY);

      if ((x2 - x1) >= 20 && (y2 - y1) >= 20) {
        candidates.push({
          x1, y1, x2, y2,
          w: x2 - x1,
          h: y2 - y1,
          score: maxScore,
          classId: bestClass
        });
      }
    }
  }

  return runNMS(candidates, iouThreshold);
}

// --- RENDERING & UI UPDATES ---
function renderDetections(source, detections, srcW, srcH) {
  displayCanvas.width = srcW;
  displayCanvas.height = srcH;

  // 1. Draw source
  displayCtx.drawImage(source, 0, 0, srcW, srcH);

  // 2. Filter skin if enabled
  const doFilterSkin = skinToggle.checked;
  const filtered = [];

  for (const d of detections) {
    if (doFilterSkin && isSkinRegion(displayCtx, d.x1, d.y1, d.x2, d.y2)) {
      continue;
    }
    filtered.push(d);
  }

  // 3. Accumulate totals
  let totalVal = 0;
  const breakdown = { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0 };

  for (const d of filtered) {
    const denom = DENOMS[d.classId];
    if (denom) {
      breakdown[d.classId]++;
      totalVal += denom.val;
    }

    // Draw box
    displayCtx.strokeStyle = denom.color;
    displayCtx.lineWidth = Math.max(3, Math.round(srcW / 250));
    displayCtx.strokeRect(d.x1, d.y1, d.w, d.h);

    // Box glow
    displayCtx.shadowColor = denom.color;
    displayCtx.shadowBlur = 10;
    displayCtx.strokeRect(d.x1, d.y1, d.w, d.h);
    displayCtx.shadowBlur = 0;

    // Label pill
    const label = `${denom.name} Birr  ${(d.score * 100).toFixed(0)}%`;
    const fontSize = Math.max(14, Math.round(srcW / 45));
    displayCtx.font = `bold ${fontSize}px "Plus Jakarta Sans", sans-serif`;
    const txtWidth = displayCtx.measureText(label).width;

    const pillH = fontSize + 10;
    const pillY = Math.max(pillH, d.y1);

    displayCtx.fillStyle = denom.color;
    displayCtx.fillRect(d.x1, pillY - pillH, txtWidth + 14, pillH);

    displayCtx.fillStyle = '#ffffff';
    displayCtx.fillText(label, d.x1 + 7, pillY - 7);
  }

  // 4. Update Dashboard metrics
  updateDashboard(totalVal, filtered.length, breakdown);

  return filtered;
}

function updateDashboard(totalVal, noteCount, breakdown) {
  totalAmount.innerText = totalVal.toLocaleString();
  totalNotesPill.innerText = `${noteCount} ${noteCount === 1 ? 'Note' : 'Notes'}`;

  // Update cards
  for (const [id, denom] of Object.entries(DENOMS)) {
    const count = breakdown[id] || 0;
    const subtotal = count * denom.val;

    const countEl = document.getElementById(`count-${denom.name}`);
    const subtotalEl = document.getElementById(`subtotal-${denom.name}`);
    const cardEl = document.getElementById(`card-${denom.name}`);

    if (countEl) countEl.innerText = count;
    if (subtotalEl) subtotalEl.innerText = `ETB ${subtotal.toLocaleString()}`;

    if (cardEl) {
      if (count > 0) {
        cardEl.classList.add('active');
      } else {
        cardEl.classList.remove('active');
      }
    }
  }

  // Auto Speech
  if (autoVoiceToggle.checked && totalVal > 0 && totalVal !== lastSpokenTotal) {
    speakAmount(totalVal, noteCount);
    lastSpokenTotal = totalVal;
  }
}

function speakAmount(val, count) {
  if (!('speechSynthesis' in window)) return;
  window.speechSynthesis.cancel();
  const text = `Total detected: ${val} Ethiopian Birr, with ${count} notes.`;
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 1.0;
  window.speechSynthesis.speak(utterance);
}

speakBtn.addEventListener('click', () => {
  const val = parseInt(totalAmount.innerText.replace(/,/g, '')) || 0;
  const count = parseInt(totalNotesPill.innerText) || 0;
  speakAmount(val, count);
});

// --- SINGLE IMAGE INFERENCE (Upload Mode) ---
async function runInferenceOnImage(img) {
  if (!ortSession) {
    alert('Model is still initializing. Please wait a few seconds.');
    return;
  }

  const tStart = performance.now();
  const tensor = preprocess(img, img.naturalWidth, img.naturalHeight);

  const feeds = { images: tensor };
  const results = await ortSession.run(feeds);
  const outputData = results[ortSession.outputNames[0]].data;

  const confThresh = parseInt(confSlider.value) / 100.0;
  const iouThresh = parseInt(iouSlider.value) / 100.0;

  const detections = postprocess(outputData, img.naturalWidth, img.naturalHeight, confThresh, iouThresh);
  renderDetections(img, detections, img.naturalWidth, img.naturalHeight);

  const tElapsed = performance.now() - tStart;
  inferTime.innerText = `${tElapsed.toFixed(1)} ms`;
}

// --- LIVE WEBCAM INFERENCE (Camera Mode) ---
async function cameraFrameLoop() {
  if (!isCameraRunning) return;

  if (!isProcessingFrame && cameraVideo.readyState === 4 && ortSession) {
    isProcessingFrame = true;
    const tStart = performance.now();

    const vw = cameraVideo.videoWidth;
    const vh = cameraVideo.videoHeight;

    if (vw > 0 && vh > 0) {
      const tensor = preprocess(cameraVideo, vw, vh);
      const feeds = { images: tensor };
      const results = await ortSession.run(feeds);
      const outputData = results[ortSession.outputNames[0]].data;

      const confThresh = parseInt(confSlider.value) / 100.0;
      const iouThresh = parseInt(iouSlider.value) / 100.0;

      const detections = postprocess(outputData, vw, vh, confThresh, iouThresh);
      renderDetections(cameraVideo, detections, vw, vh);

      const tElapsed = performance.now() - tStart;
      inferTime.innerText = `${tElapsed.toFixed(1)} ms`;

      // Calculate FPS
      frameCount++;
      const now = performance.now();
      if (now - lastFpsTime >= 1000) {
        fpsVal.innerText = frameCount;
        frameCount = 0;
        lastFpsTime = now;
      }
    }
    isProcessingFrame = false;
  }

  cameraAnimationId = requestAnimationFrame(cameraFrameLoop);
}

async function startCamera() {
  try {
    if (cameraStream) {
      cameraStream.getTracks().forEach(t => t.stop());
    }

    const constraints = {
      video: {
        facingMode: cameraFacingMode,
        width: { ideal: 1280 },
        height: { ideal: 720 }
      },
      audio: false
    };

    cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
    cameraVideo.srcObject = cameraStream;
    await cameraVideo.play();

    isCameraRunning = true;
    startCamBtn.innerHTML = '<span>⏸</span> Stop Camera';
    startCamBtn.classList.remove('btn-primary');

    cameraFrameLoop();
  } catch (err) {
    console.error('Camera error:', err);
    alert('Unable to access camera: ' + err.message);
  }
}

function stopCamera() {
  if (cameraAnimationId) {
    cancelAnimationFrame(cameraAnimationId);
    cameraAnimationId = null;
  }
  if (cameraStream) {
    cameraStream.getTracks().forEach(t => t.stop());
    cameraStream = null;
  }
  isCameraRunning = false;
  startCamBtn.innerHTML = '<span>▶</span> Start Camera';
  startCamBtn.classList.add('btn-primary');
  fpsVal.innerText = '0';
}

startCamBtn.addEventListener('click', () => {
  if (isCameraRunning) {
    stopCamera();
  } else {
    startCamera();
  }
});

switchCamBtn.addEventListener('click', () => {
  cameraFacingMode = cameraFacingMode === 'environment' ? 'user' : 'environment';
  if (isCameraRunning) {
    startCamera();
  }
});

// --- TAB SWITCHING ---
tabUploadBtn.addEventListener('click', () => {
  activeMode = 'upload';
  tabUploadBtn.classList.add('active');
  tabCameraBtn.classList.remove('active');

  stopCamera();
  startCamBtn.style.display = 'none';
  switchCamBtn.style.display = 'none';
  chooseImageBtn.style.display = 'inline-flex';

  if (currentImage) {
    dropzone.style.display = 'none';
    runInferenceOnImage(currentImage);
  } else {
    dropzone.style.display = 'flex';
  }
});

tabCameraBtn.addEventListener('click', () => {
  activeMode = 'camera';
  tabCameraBtn.classList.add('active');
  tabUploadBtn.classList.remove('active');

  dropzone.style.display = 'none';
  chooseImageBtn.style.display = 'none';
  startCamBtn.style.display = 'inline-flex';
  switchCamBtn.style.display = 'inline-flex';

  if (!isCameraRunning) {
    startCamera();
  }
});

// --- UPLOAD HANDLERS ---
chooseImageBtn.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('click', () => fileInput.click());

dropzone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropzone.classList.add('dragover');
});

dropzone.addEventListener('dragleave', () => {
  dropzone.classList.remove('dragover');
});

dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropzone.classList.remove('dragover');
  if (e.dataTransfer.files.length > 0) {
    loadSelectedFile(e.dataTransfer.files[0]);
  }
});

fileInput.addEventListener('change', (e) => {
  if (e.target.files.length > 0) {
    loadSelectedFile(e.target.files[0]);
  }
});

function loadSelectedFile(file) {
  if (!file.type.startsWith('image/')) {
    alert('Please select an image file (.jpg, .png, etc.)');
    return;
  }

  const reader = new FileReader();
  reader.onload = (event) => {
    const img = new Image();
    img.onload = () => {
      currentImage = img;
      dropzone.style.display = 'none';
      runInferenceOnImage(img);
    };
    img.src = event.target.result;
  };
  reader.readAsDataURL(file);
}

// Slider updates
confSlider.addEventListener('input', (e) => {
  confVal.innerText = `${e.target.value}%`;
  if (activeMode === 'upload' && currentImage) {
    runInferenceOnImage(currentImage);
  }
});

iouSlider.addEventListener('input', (e) => {
  iouVal.innerText = `${e.target.value}%`;
  if (activeMode === 'upload' && currentImage) {
    runInferenceOnImage(currentImage);
  }
});

skinToggle.addEventListener('change', () => {
  if (activeMode === 'upload' && currentImage) {
    runInferenceOnImage(currentImage);
  }
});

// Download button
downloadBtn.addEventListener('click', () => {
  if (!displayCanvas || displayCanvas.width === 0) {
    alert('No detection to download!');
    return;
  }
  const link = document.createElement('a');
  link.download = `ethiopian_birr_detection_${Date.now()}.jpg`;
  link.href = displayCanvas.toDataURL('image/jpeg', 0.95);
  link.click();
});

// Paste from clipboard support
window.addEventListener('paste', (e) => {
  const items = e.clipboardData?.items;
  if (!items) return;
  for (const item of items) {
    if (item.type.startsWith('image/')) {
      const file = item.getAsFile();
      if (file) {
        if (activeMode !== 'upload') {
          tabUploadBtn.click();
        }
        loadSelectedFile(file);
        break;
      }
    }
  }
});

// Initialize on page load
window.addEventListener('DOMContentLoaded', () => {
  initModel();
});
