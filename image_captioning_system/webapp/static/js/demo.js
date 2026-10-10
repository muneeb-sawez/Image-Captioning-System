/**
 * Caption Lens — Interactive Demo Controller
 * Manages file uploads, drag-drop, sample selectors, multi-step progress, and dual inference.
 */

document.addEventListener("DOMContentLoaded", () => {
  const API_URL = "/api/caption";

  let activeFile = null;
  let activeSample = null;
  let objectUrl = null;

  // DOM Elements
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const dropzonePrompt = document.getElementById("dropzonePrompt");
  const previewFrame = document.getElementById("previewFrame");
  const imagePreview = document.getElementById("imagePreview");
  const sampleButtons = document.querySelectorAll(".sample-btn, .sample-chip");

  const beamSlider = document.getElementById("beamSlider");
  const beamVal = document.getElementById("beamVal");
  const beamDesc = document.getElementById("beamDesc");

  const btnGenerate = document.getElementById("btnGenerate");
  const btnReset = document.getElementById("btnReset");

  const statusBar = document.getElementById("statusBar");
  const statusDot = document.getElementById("statusDot");
  const statusText = document.getElementById("statusText");
  const errorBanner = document.getElementById("errorBanner");

  const captionBlip = document.getElementById("captionBlip");
  const captionVit = document.getElementById("captionVit");
  const captionGru = document.getElementById("captionGru");
  const timingBlip = document.getElementById("timingBlip");
  const timingVit = document.getElementById("timingVit");
  const timingGru = document.getElementById("timingGru");
  const copyButtons = document.querySelectorAll(".copy-btn");

  // Beam Slider Listener
  if (beamSlider) {
    beamSlider.addEventListener("input", (e) => {
      const v = parseInt(e.target.value, 10);
      beamVal.textContent = v;
      if (v === 1) {
        beamDesc.textContent = "Greedy argmax decoding (fastest)";
      } else if (v <= 4) {
        beamDesc.textContent = "Fast constrained beam search";
      } else if (v <= 7) {
        beamDesc.textContent = "Standard balanced beam search";
      } else {
        beamDesc.textContent = "Exhaustive candidate hypothesis search";
      }
    });
  }

  // Dropzone Events
  if (dropzone && fileInput) {
    dropzone.addEventListener("click", () => fileInput.click());
    dropzone.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        fileInput.click();
      }
    });

    ["dragenter", "dragover"].forEach((evt) => {
      dropzone.addEventListener(evt, (e) => {
        e.preventDefault();
        dropzone.classList.add("drag-active");
      });
    });

    ["dragleave", "drop"].forEach((evt) => {
      dropzone.addEventListener(evt, (e) => {
        e.preventDefault();
        dropzone.classList.remove("drag-active");
      });
    });

    dropzone.addEventListener("drop", (e) => {
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFileSelect(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files[0]) {
        handleFileSelect(e.target.files[0]);
      }
    });
  }

  // Sample Buttons Listener
  sampleButtons.forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const sampleName = btn.dataset.sample;
      if (!sampleName) return;

      activeFile = null;
      activeSample = sampleName;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      objectUrl = null;
      if (fileInput) fileInput.value = "";

      imagePreview.src = `/static/samples/${sampleName}`;
      dropzonePrompt.classList.add("d-none");
      previewFrame.classList.remove("d-none");

      btnGenerate.disabled = false;
      btnReset.disabled = false;
      clearOutputs();
      setStatus(`Sample loaded (${btn.textContent.trim()}). Click "Generate Captions" to run models.`, false);
    });
  });

  function handleFileSelect(f) {
    if (!f) return;
    if (!/^image\/(jpeg|png|webp)$/i.test(f.type)) {
      showError("Unsupported file format. Please upload a JPG, PNG, or WebP photo.");
      return;
    }
    if (f.size > 15 * 1024 * 1024) {
      showError("File size exceeds 15 MB. Please select a smaller photo.");
      return;
    }

    activeFile = f;
    activeSample = null;
    hideError();

    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = URL.createObjectURL(f);

    imagePreview.src = objectUrl;
    dropzonePrompt.classList.add("d-none");
    previewFrame.classList.remove("d-none");

    btnGenerate.disabled = false;
    btnReset.disabled = false;
    clearOutputs();
    setStatus("Image loaded. Click 'Generate Captions' to analyze.", false);
  }

  function setStatus(text, isActive = false) {
    statusText.textContent = text;
    if (isActive) {
      statusDot.classList.add("active");
    } else {
      statusDot.classList.remove("active");
    }
  }

  function showError(msg) {
    errorBanner.textContent = msg;
    errorBanner.classList.remove("d-none");
  }

  function hideError() {
    errorBanner.classList.add("d-none");
    errorBanner.textContent = "";
  }

  function clearOutputs() {
    captionBlip.classList.add("empty");
    captionBlip.textContent = "Generated BLIP caption will appear here...";
    if (captionVit) {
      captionVit.classList.add("empty");
      captionVit.textContent = "Vision Transformer caption will appear here...";
    }
    captionGru.classList.add("empty");
    captionGru.textContent = "Custom GRU caption will appear here...";
    timingBlip.textContent = "Latency: — ms";
    if (timingVit) timingVit.textContent = "Latency: — ms";
    timingGru.textContent = "Latency: — ms";
    hideError();
  }

  // Reset Handler
  if (btnReset) {
    btnReset.addEventListener("click", () => {
      activeFile = null;
      activeSample = null;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      objectUrl = null;
      if (fileInput) fileInput.value = "";

      imagePreview.removeAttribute("src");
      previewFrame.classList.add("d-none");
      dropzonePrompt.classList.remove("d-none");

      btnGenerate.disabled = true;
      btnReset.disabled = true;
      clearOutputs();
      setStatus("Select or upload an image to begin generation.", false);
    });
  }

  // Generate Handler
  if (btnGenerate) {
    btnGenerate.addEventListener("click", async () => {
      if (!activeFile && !activeSample) return;

      btnGenerate.disabled = true;
      btnReset.disabled = true;
      hideError();
      clearOutputs();

      captionBlip.innerHTML = '<span class="text-tertiary">Synthesizing multimodal attention...</span>';
      if (captionVit) captionVit.innerHTML = '<span class="text-tertiary">Cross-attending visual patches with GPT-2...</span>';
      captionGru.innerHTML = '<span class="text-tertiary">Decoding recurrent sequence...</span>';

      const pipelineStages = [
        "Analyzing image geometry and color tensors...",
        "Cross-attending patch tokens with Vision Transformer...",
        "Extracting 2048-d visual features with ResNet-50...",
        "Generating token hypotheses with Beam Search...",
        "Evaluating Salesforce BLIP vision-language baseline..."
      ];

      let stageIdx = 0;
      setStatus(pipelineStages[0], true);
      const stageTimer = setInterval(() => {
        stageIdx = (stageIdx + 1) % pipelineStages.length;
        setStatus(pipelineStages[stageIdx], true);
      }, 750);

      try {
        const formData = new FormData();
        if (activeFile) {
          formData.append("image", activeFile);
        } else if (activeSample) {
          formData.append("sample", activeSample);
        }
        formData.append("beams", beamSlider.value);

        const response = await fetch(API_URL, {
          method: "POST",
          body: formData
        });

        clearInterval(stageTimer);

        if (!response.ok) {
          const errData = await response.json().catch(() => ({}));
          throw new Error(errData.error || `Server responded with status ${response.status}`);
        }

        const data = await response.json();

        // Render BLIP Result
        captionBlip.classList.remove("empty");
        captionBlip.textContent = data.blip || "No BLIP caption returned.";
        if (data.timing && data.timing.blip_ms) {
          timingBlip.textContent = `Latency: ${data.timing.blip_ms} ms • ViT-B/16 + Multimodal Decoders`;
          timingBlip.innerHTML = `Latency: ${data.timing.blip_ms} ms • ViT-B/16 + Multimodal Decoders`;
        }

        // Render ViT-GPT2 Result
        if (captionVit) {
          captionVit.classList.remove("empty");
          captionVit.textContent = data.vit || "No Vision Transformer caption returned.";
          if (data.timing && data.timing.vit_ms && timingVit) {
            timingVit.textContent = `Latency: ${data.timing.vit_ms} ms • ViT-B/16 + GPT-2 Cross-Attention`;
            timingVit.innerHTML = `Latency: ${data.timing.vit_ms} ms • ViT-B/16 + GPT-2 Cross-Attention`;
          }
        }

        // Render GRU Result
        captionGru.classList.remove("empty");
        captionGru.textContent = data.gru || "No GRU caption returned.";
        if (data.timing && data.timing.gru_ms) {
          timingGru.textContent = `Latency: ${data.timing.gru_ms} ms • ResNet-50 + 2-Layer GRU (k=${beamSlider.value})`;
          timingGru.innerHTML = `Latency: ${data.timing.gru_ms} ms • ResNet-50 + 2-Layer GRU (k=${beamSlider.value})`;
        }

        setStatus(`Inference complete (${data.timing ? data.timing.total_ms : 0} ms total). Review captions below.`, false);
      } catch (err) {
        clearInterval(stageTimer);
        showError(err.message || "An unexpected error occurred during model generation.");
        setStatus("Inference encountered an error. Check details above.", false);
        captionBlip.classList.add("empty");
        captionBlip.textContent = "Error running BLIP.";
        if (captionVit) {
          captionVit.classList.add("empty");
          captionVit.textContent = "Error running Vision Transformer.";
        }
        captionGru.classList.add("empty");
        captionGru.textContent = "Error running Custom GRU.";
      } finally {
        btnGenerate.disabled = false;
        btnReset.disabled = false;
      }
    });
  }

  // Copy Buttons
  copyButtons.forEach((btn) => {
    btn.addEventListener("click", async () => {
      const targetId = btn.dataset.target;
      const targetEl = document.getElementById(targetId);
      if (!targetEl || targetEl.classList.contains("empty")) return;

      try {
        await navigator.clipboard.writeText(targetEl.textContent.trim());
        const origHtml = btn.innerHTML;
        btn.innerHTML = '<i class="bi bi-check2 text-cyan"></i>';
        setTimeout(() => {
          btn.innerHTML = origHtml;
        }, 1500);
      } catch (err) {
        console.error("Copy failed", err);
      }
    });
  });
});
