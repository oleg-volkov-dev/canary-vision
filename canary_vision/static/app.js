"use strict";

const $ = (id) => document.getElementById(id);
const state = { file: null, objectUrl: null, ready: false, busy: false, result: null, selection: 0, request: null, samples: [], model: null };

function syncButton() {
  $("run-inference").disabled = !state.file || !state.ready || state.busy;
  $("run-label").textContent = state.busy ? "Analyzing image…" : "Run inference";
  $("results-panel").setAttribute("aria-busy", String(state.busy));
}

function showError(message) {
  $("error-message").textContent = message;
  $("error-message").hidden = !message;
}

function resultView(view) {
  $("result-empty").hidden = view !== "empty";
  $("result-loading").hidden = view !== "loading";
  $("prediction-content").hidden = view !== "predictions";
  $("waiting-text").textContent = state.file ? "READY WHEN YOU ARE" : "WAITING FOR AN IMAGE";
}

function resetSelection() {
  state.selection += 1;
  state.request?.abort();
  state.request = null;
  state.file = null;
  state.result = null;
  state.busy = false;
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  state.objectUrl = null;
  $("preview").removeAttribute("src");
  $("preview-wrap").hidden = true;
  $("dropzone-empty").hidden = false;
  $("file-meta").hidden = true;
  $("file-input").value = "";
  document.querySelectorAll(".sample").forEach((button) => button.classList.remove("selected"));
  showError("");
  resultView("empty");
  syncButton();
}

async function selectFile(file, sampleId = null) {
  resetSelection();
  if (!file) return;
  if (file.size > 10 * 1024 * 1024) return showError("That image is a little too big. Choose a file of 10 MB or smaller.");
  if (file.size === 0) return showError("That file is empty. Choose a JPEG, PNG, or WebP image.");
  if (file.type && !["image/jpeg", "image/png", "image/webp"].includes(file.type)) return showError("Choose a JPEG, PNG, or WebP image. Other formats aren’t supported yet.");
  const selection = state.selection;
  const objectUrl = URL.createObjectURL(file);
  state.objectUrl = objectUrl;
  const image = new Image();
  image.src = objectUrl;
  try {
    await image.decode();
    if (selection !== state.selection) return;
    if (image.naturalWidth * image.naturalHeight > 20_000_000) {
      resetSelection();
      return showError("Choose an image with 20 megapixels or fewer.");
    }
    state.file = file;
    $("preview").src = objectUrl;
    $("preview").alt = `Preview of ${file.name}`;
    $("preview-dimensions").textContent = `${image.naturalWidth} × ${image.naturalHeight}`;
    $("preview-wrap").hidden = false;
    $("dropzone-empty").hidden = true;
    $("file-name").textContent = file.name;
    $("file-size").textContent = file.size >= 1024 * 1024 ? `${(file.size / (1024 * 1024)).toFixed(1)} MB` : `${Math.max(1, Math.round(file.size / 1024))} KB`;
    $("file-meta").hidden = false;
    if (sampleId) document.querySelector(`[data-sample="${sampleId}"]`)?.classList.add("selected");
    resultView("empty");
    syncButton();
  } catch {
    if (selection !== state.selection) return;
    resetSelection();
    showError("We couldn’t open that image. Try another JPEG, PNG, or WebP file.");
  }
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  let body;
  try { body = await response.json(); } catch { throw new Error("The service returned an unexpected response. Please try again."); }
  if (!response.ok) throw new Error(body.error?.message || "The request could not be completed.");
  return body;
}

function renderPredictions(result) {
  $("predictions").replaceChildren();
  result.predictions.forEach((prediction, index) => {
    const li = document.createElement("li");
    li.className = "prediction";
    const row = document.createElement("div");
    row.className = "prediction-row";
    const rank = document.createElement("span");
    rank.className = "prediction-rank";
    rank.textContent = String(index + 1).padStart(2, "0");
    const label = document.createElement("span");
    label.className = "prediction-label";
    label.textContent = prediction.label;
    const score = document.createElement("span");
    score.className = "prediction-score";
    score.textContent = (prediction.score * 100).toFixed(1);
    const unit = document.createElement("span");
    unit.className = "score-unit";
    unit.textContent = "%";
    score.append(unit);
    row.append(rank, label, score);
    const track = document.createElement("div");
    track.className = "score-track";
    track.setAttribute("aria-hidden", "true");
    const fill = document.createElement("div");
    fill.className = "score-fill";
    fill.style.width = `${prediction.score * 100}%`;
    track.append(fill);
    li.append(row, track);
    $("predictions").append(li);
  });
  $("inference-time").textContent = `${result.inference_ms.toFixed(1)} ms`;
  $("model-version").textContent = result.model_version;
  resultView("predictions");
}

async function runInference() {
  if (!state.file || !state.ready || state.busy) return;
  showError("");
  state.result = null;
  state.busy = true;
  const selection = state.selection;
  const controller = new AbortController();
  state.request = controller;
  const timeout = setTimeout(() => controller.abort("timeout"), 60_000);
  syncButton();
  resultView("loading");
  const form = new FormData();
  form.append("file", state.file);
  try {
    const result = await fetchJson("/predict", { method: "POST", body: form, signal: controller.signal });
    if (selection !== state.selection) return;
    state.result = result;
    renderPredictions(result);
  } catch (error) {
    if (selection !== state.selection) return;
    resultView("empty");
    showError(controller.signal.aborted ? "Inference timed out. Please try again." : error.message === "Failed to fetch" ? "The service couldn’t be reached. Check that it’s running and try again." : error.message);
  } finally {
    clearTimeout(timeout);
    if (selection === state.selection) {
      state.busy = false;
      state.request = null;
      syncButton();
    }
  }
}

async function selectSample(id) {
  resetSelection();
  const selection = state.selection;
  const controller = new AbortController();
  state.request = controller;
  $("waiting-text").textContent = "LOADING SAMPLE IMAGE";
  const timeout = setTimeout(() => controller.abort(), 15_000);
  try {
    const response = await fetch(`/samples/${id}`, { signal: controller.signal });
    if (!response.ok) throw new Error("The sample image is unavailable. Try uploading your own.");
    const blob = await response.blob();
    if (selection !== state.selection) return;
    await selectFile(new File([blob], `${id}.png`, { type: "image/png" }), id);
  } catch (error) {
    if (selection !== state.selection) return;
    resultView("empty");
    showError(controller.signal.aborted ? "The sample took too long to load. Please try again." : error.message);
  } finally {
    clearTimeout(timeout);
  }
}

async function checkService() {
  try {
    const result = await fetchJson("/ready", { signal: AbortSignal.timeout(5000) });
    state.ready = result.status === "ready";
  } catch { state.ready = false; }
  [$("service-dot"), $("sidebar-dot")].forEach((dot) => {
    dot.classList.remove("muted");
    dot.classList.toggle("offline", !state.ready);
  });
  $("service-status").textContent = state.ready ? "Service ready" : "Service unavailable";
  $("sidebar-status").textContent = state.ready ? "Local environment ready" : "Service unavailable";
  syncButton();
}

function detailGrid(items) {
  const grid = document.createElement("dl");
  grid.className = "detail-grid";
  items.forEach(([name, value]) => {
    const cell = document.createElement("div");
    const term = document.createElement("dt");
    term.textContent = name;
    const definition = document.createElement("dd");
    definition.textContent = value;
    cell.append(term, definition);
    grid.append(cell);
  });
  return grid;
}

function paragraph(text) { const p = document.createElement("p"); p.textContent = text; return p; }

async function openDialog(kind) {
  const dialog = $("info-dialog");
  $("dialog-content").replaceChildren(paragraph("Loading…"));
  $("dialog-eyebrow").textContent = kind === "model" ? "MODEL CARD" : "RECORDED REGRESSION CHECK";
  $("dialog-title").textContent = kind === "model" ? "MobileNetV3 Small" : "A small test. A useful signal.";
  if (!dialog.open) dialog.showModal();
  dialog.dataset.kind = kind;
  try {
    const data = await fetchJson(kind === "model" ? "/model" : "/evaluation", { signal: AbortSignal.timeout(5000) });
    if (dialog.dataset.kind !== kind) return;
    const content = $("dialog-content");
    content.replaceChildren();
    if (kind === "model") {
      content.append(paragraph("A lightweight image classifier with pinned, pretrained ImageNet weights. The image is resized to 256 pixels, center-cropped to 224 × 224, and normalized using the weights’ prescribed transforms."));
      content.append(detailGrid([["WEIGHTS", data.weights], ["COMPUTE", data.device.toUpperCase()], ["CLASSES", data.classes.toLocaleString()], ["PARAMETERS", data.parameters.toLocaleString()], ["INPUT", `${data.input_size.join(" × ")} pixels`], ["UPLOAD LIMIT", "10 MB / 20 megapixels"]]));
      const version = document.createElement("code");
      version.className = "dialog-code";
      version.textContent = `${data.model_version}\nSHA-256: ${data.weights_sha256}`;
      content.append(version, paragraph("This model recognizes ImageNet categories. It may confidently misclassify images outside those categories, abstract images, or scenes with several objects. Scores are not calibrated certainty."));
      const link = document.createElement("a");
      link.href = "https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html";
      link.target = "_blank";
      link.rel = "noopener";
      link.textContent = "Read the Torchvision model documentation ↗";
      content.append(link);
      content.append(paragraph("Sample photography: Chelsea — Stefan van der Walt; coffee — Rachel Michetti; camera — Lav Varshney (CC0). The clock is public domain by Stefan van der Walt. Permissions are recorded in the repository’s evaluation manifest."));
    } else {
      content.append(paragraph(`Recorded results for ${data.dataset}: ${data.sample_count} fixed, redistributable images. This report is bundled with the app; opening this panel does not rerun evaluation.`));
      content.append(detailGrid([["TOP-1 ACCURACY", `${(data.top1_accuracy * 100).toFixed(0)}%`], ["TOP-3 ACCURACY", `${(data.top3_accuracy * 100).toFixed(0)}%`], ["REQUIRED TOP-1", `${(data.min_top1_accuracy * 100).toFixed(0)}%`], ["GATE", data.passed ? "Passed" : "Failed"]]));
      const table = document.createElement("table");
      table.className = "evaluation-table";
      const head = table.createTHead().insertRow();
      ["SAMPLE", "TOP PREDICTION", "TOP-1"].forEach((text) => { const th = document.createElement("th"); th.textContent = text; head.append(th); });
      const body = table.createTBody();
      data.results.forEach((sample) => {
        const row = body.insertRow();
        [sample.id, sample.predictions[0].label, sample.top1_correct ? "Pass" : "Miss"].forEach((text) => { const cell = row.insertCell(); cell.textContent = text; });
        if (!sample.top1_correct) row.className = "fail";
      });
      content.append(table, paragraph("This four-image check detects obvious regressions. It is too small to estimate real-world accuracy. Human labels, accepted categories, image checksums, and redistribution permissions are fixed in the evaluation manifest."));
      const version = document.createElement("code");
      version.className = "dialog-code";
      version.textContent = data.model_version;
      content.append(version);
    }
  } catch (error) {
    if (dialog.dataset.kind === kind) $("dialog-content").replaceChildren(paragraph(error.message));
  }
}

$("dropzone").addEventListener("click", () => $("file-input").click());
$("file-input").addEventListener("change", (event) => selectFile(event.target.files[0]));
$("clear-image").addEventListener("click", resetSelection);
$("run-inference").addEventListener("click", runInference);
document.querySelectorAll("[data-sample]").forEach((button) => button.addEventListener("click", () => selectSample(button.dataset.sample)));
document.querySelectorAll("[data-dialog]").forEach((button) => button.addEventListener("click", () => openDialog(button.dataset.dialog)));
$("close-dialog").addEventListener("click", () => $("info-dialog").close());
$("info-dialog").addEventListener("click", (event) => { const rect = $("info-dialog").getBoundingClientRect(); if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) $("info-dialog").close(); });
["dragenter", "dragover"].forEach((event) => $("dropzone").addEventListener(event, (e) => { e.preventDefault(); $("dropzone").classList.add("dragging"); }));
["dragleave", "drop"].forEach((event) => $("dropzone").addEventListener(event, (e) => { e.preventDefault(); $("dropzone").classList.remove("dragging"); }));
$("dropzone").addEventListener("drop", (event) => selectFile(event.dataTransfer.files[0]));
// Prevent a dropped file outside the upload area from navigating away from the app.
window.addEventListener("dragover", (event) => event.preventDefault());
window.addEventListener("drop", (event) => event.preventDefault());
document.addEventListener("keydown", (event) => { if (event.key === "Enter" && !$("info-dialog").open && !["BUTTON", "A", "INPUT"].includes(document.activeElement.tagName)) { event.preventDefault(); runInference(); } });
$("export-json").addEventListener("click", () => {
  if (!state.result) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(state.result, null, 2)], { type: "application/json" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = `canary-${state.result.request_id}.json`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

checkService();
setInterval(checkService, 15_000);
fetchJson("/model").then((model) => { state.model = model; $("app-version").textContent = model.app_version; }).catch(() => {});
