// ==========================================================
// CIFAR-10 AI Vision — Client-Side Application Logic
// Vanilla JavaScript • Async Fetch • Drag & Drop
// ==========================================================

// DOM Elements
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const resultsEmpty = document.getElementById('results-empty');
const resultsActive = document.getElementById('results-active');
const loadingOverlay = document.getElementById('loading-overlay');
const previewImage = document.getElementById('preview-image');
const predIcon = document.getElementById('pred-icon');
const predClass = document.getElementById('pred-class');
const predConfidence = document.getElementById('pred-confidence');
const probabilitiesList = document.getElementById('probabilities-list');
const latencyTag = document.getElementById('latency-tag');
const latencyMs = document.getElementById('latency-ms');

// Modal Elements
const imageModal = document.getElementById('image-modal');
const modalImg = document.getElementById('modal-img');
const modalTitle = document.getElementById('modal-title');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
  setupDropzone();
  setupClipboardPaste();
});

// ----------------------------------------------------------
// Tab Navigation
// ----------------------------------------------------------
function switchTab(tabName) {
  // Update Buttons
  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
  const activeBtn = document.getElementById(`tab-${tabName}`);
  if (activeBtn) activeBtn.classList.add('active');

  // Update Views
  const views = ['classifier', 'visualizations', 'architecture'];
  views.forEach(v => {
    const el = document.getElementById(`view-${v}`);
    if (el) {
      el.style.display = (v === tabName) ? 'block' : 'none';
      if (v === tabName) el.classList.add('active');
      else el.classList.remove('active');
    }
  });
}

// ----------------------------------------------------------
// Drag & Drop / File Input Handling
// ----------------------------------------------------------
function setupDropzone() {
  if (!dropzone || !fileInput) return;

  // Open file picker on click
  dropzone.addEventListener('click', (e) => {
    // Only if not clicking browse link (which handles itself)
    if (!e.target.classList.contains('browse-link')) {
      fileInput.click();
    }
  });

  // Drag over / leave
  ['dragenter', 'dragover'].forEach(name => {
    dropzone.addEventListener(name, (e) => {
      e.preventDefault();
      dropzone.classList.add('drag-over');
    });
  });

  ['dragleave', 'drop'].forEach(name => {
    dropzone.addEventListener(name, (e) => {
      e.preventDefault();
      dropzone.classList.remove('drag-over');
    });
  });

  // Handle dropped files
  dropzone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type.startsWith('image/')) {
      uploadAndClassify(files[0]);
    }
  });

  // Handle file input change
  fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
      uploadAndClassify(fileInput.files[0]);
    }
  });
}

// Support Ctrl+V paste from clipboard
function setupClipboardPaste() {
  window.addEventListener('paste', (e) => {
    const items = (e.clipboardData || e.originalEvent.clipboardData).items;
    for (const item of items) {
      if (item.type.indexOf('image') !== -1) {
        const file = item.getAsFile();
        uploadAndClassify(file);
        break;
      }
    }
  });
}

// ----------------------------------------------------------
// API: Upload and Classify Custom Image
// ----------------------------------------------------------
async function uploadAndClassify(file) {
  // Show image preview immediately
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImage.src = e.target.result;
  };
  reader.readAsDataURL(file);

  showLoading(true);

  const formData = new FormData();
  formData.append('image', file);

  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      body: formData
    });

    const data = await response.json();
    if (data.success) {
      displayResults(data);
    } else {
      alert('Error predicting image: ' + (data.error || 'Unknown error'));
      showLoading(false);
    }
  } catch (err) {
    console.error(err);
    alert('Server communication error. Check if backend is running.');
    showLoading(false);
  }
}

// ----------------------------------------------------------
// API: Load and Classify Pre-Saved Sample
// ----------------------------------------------------------
async function loadSample(sampleName) {
  // Update preview image to sample thumbnail
  previewImage.src = `/static/samples/${sampleName.toLowerCase()}.png`;

  showLoading(true);

  try {
    const response = await fetch(`/api/predict-sample/${sampleName.toLowerCase()}`);
    const data = await response.json();

    if (data.success) {
      displayResults(data);
    } else {
      alert('Error predicting sample: ' + (data.error || 'Unknown error'));
      showLoading(false);
    }
  } catch (err) {
    console.error(err);
    alert('Failed to connect to backend server.');
    showLoading(false);
  }
}

// ----------------------------------------------------------
// UI: Render Results
// ----------------------------------------------------------
function displayResults(data) {
  showLoading(false);

  resultsEmpty.style.display = 'none';
  resultsActive.style.display = 'flex';

  // Hero Info
  predIcon.textContent = data.icon;
  predClass.textContent = data.prediction;
  predConfidence.textContent = `${data.confidence.toFixed(1)}%`;

  // Latency Tag
  if (data.inference_time_ms !== undefined) {
    latencyTag.style.display = 'inline-flex';
    latencyMs.textContent = data.inference_time_ms;
  }

  // Populate Probabilities Breakdown
  probabilitiesList.innerHTML = '';
  data.breakdown.forEach(item => {
    const row = document.createElement('div');
    row.className = `prob-row ${item.is_top ? 'is-top' : ''}`;

    const label = document.createElement('div');
    label.className = 'prob-label';
    label.innerHTML = `<span>${item.icon}</span> <span>${item.class}</span>`;

    const barContainer = document.createElement('div');
    barContainer.className = 'prob-bar-container';

    const barFill = document.createElement('div');
    barFill.className = 'prob-bar-fill';
    // Initial 0% for animation
    barFill.style.width = '0%';

    const pct = document.createElement('div');
    pct.className = 'prob-pct';
    pct.textContent = item.percentage;

    barContainer.appendChild(barFill);
    row.appendChild(label);
    row.appendChild(barContainer);
    row.appendChild(pct);

    probabilitiesList.appendChild(row);

    // Trigger smooth fill animation
    setTimeout(() => {
      barFill.style.width = `${Math.max(item.probability * 100, 1)}%`;
    }, 50);
  });
}

function showLoading(isLoading) {
  if (loadingOverlay) {
    loadingOverlay.style.display = isLoading ? 'flex' : 'none';
  }
}

// ----------------------------------------------------------
// Lightbox Modal for Graphs
// ----------------------------------------------------------
function openModal(imgSrc, title) {
  if (!imageModal || !modalImg) return;
  modalImg.src = imgSrc;
  modalTitle.textContent = title || 'Visualization Viewer';
  imageModal.classList.add('active');
  document.body.style.overflow = 'hidden';
}

function closeModal() {
  if (!imageModal) return;
  imageModal.classList.remove('active');
  document.body.style.overflow = '';
}

// Close modal on Escape key
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') closeModal();
});
