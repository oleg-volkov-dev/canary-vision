"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  file: null,
  objectUrl: null,
  ready: false,
  busy: false,
  result: null,
  selection: 0,
  request: null,
};

function syncButton() {
  $("run-inference").disabled = !state.file || !state.ready || state.busy;
  $("run-label").textContent = state.busy
    ? "Analyzing image…"
    : "Run inference";
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
  $("waiting-text").textContent = state.file
    ? "READY WHEN YOU ARE"
    : "WAITING FOR AN IMAGE";
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
  document
    .querySelectorAll(".sample")
    .forEach((button) => button.classList.remove("selected"));
  showError("");
  resultView("empty");
  syncButton();
}

async function selectFile(file, sampleId = null) {
  resetSelection();
  if (!file) return;
  if (file.size > 10 * 1024 * 1024)
    return showError("Choose an image of 10 MB or smaller.");
  if (file.size === 0)
    return showError("That file is empty. Choose a JPEG, PNG, or WebP image.");
  if (
    file.type &&
    !["image/jpeg", "image/png", "image/webp"].includes(file.type)
  )
    return showError(
      "Choose a JPEG, PNG, or WebP image. Other formats aren’t supported yet.",
    );
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
    $("preview-dimensions").textContent =
      `${image.naturalWidth} × ${image.naturalHeight}`;
    $("preview-wrap").hidden = false;
    $("dropzone-empty").hidden = true;
    $("file-name").textContent = file.name;
    $("file-size").textContent =
      file.size >= 1024 * 1024
        ? `${(file.size / (1024 * 1024)).toFixed(1)} MB`
        : `${Math.max(1, Math.round(file.size / 1024))} KB`;
    $("file-meta").hidden = false;
    if (sampleId)
      document
        .querySelector(`[data-sample="${sampleId}"]`)
        ?.classList.add("selected");
    resultView("empty");
    syncButton();
  } catch {
    if (selection !== state.selection) return;
    resetSelection();
    showError(
      "We couldn’t open that image. Try another JPEG, PNG, or WebP file.",
    );
  }
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  let body;
  try {
    body = await response.json();
  } catch {
    throw new Error(
      "The service returned an unexpected response. Please try again.",
    );
  }
  if (!response.ok)
    throw new Error(
      body.error?.message || "The request could not be completed.",
    );
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
    const result = await fetchJson("/predict", {
      method: "POST",
      body: form,
      signal: controller.signal,
    });
    if (selection !== state.selection) return;
    state.result = result;
    renderPredictions(result);
  } catch (error) {
    if (selection !== state.selection) return;
    resultView("empty");
    showError(
      controller.signal.aborted
        ? "Inference timed out. Please try again."
        : error.message === "Failed to fetch"
          ? "The service couldn’t be reached. Check that it’s running and try again."
          : error.message,
    );
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
    const response = await fetch(`/samples/${id}`, {
      signal: controller.signal,
    });
    if (!response.ok)
      throw new Error(
        "The sample image is unavailable. Try uploading your own.",
      );
    const blob = await response.blob();
    if (selection !== state.selection) return;
    await selectFile(new File([blob], `${id}.png`, { type: "image/png" }), id);
  } catch (error) {
    if (selection !== state.selection) return;
    resultView("empty");
    showError(
      controller.signal.aborted
        ? "The sample took too long to load. Please try again."
        : error.message,
    );
  } finally {
    clearTimeout(timeout);
  }
}

async function checkService() {
  try {
    const result = await fetchJson("/ready", {
      signal: AbortSignal.timeout(5000),
    });
    state.ready = result.status === "ready";
  } catch {
    state.ready = false;
  }
  [$("service-dot"), $("sidebar-dot")].forEach((dot) => {
    dot.classList.remove("muted");
    dot.classList.toggle("offline", !state.ready);
  });
  $("service-status").textContent = state.ready
    ? "Service ready"
    : "Service unavailable";
  $("sidebar-status").textContent = state.ready
    ? "Local environment ready"
    : "Service unavailable";
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

function paragraph(text) {
  const p = document.createElement("p");
  p.textContent = text;
  return p;
}

async function openDialog(kind) {
  const dialog = $("info-dialog");
  $("dialog-content").replaceChildren(paragraph("Loading…"));
  $("dialog-eyebrow").textContent =
    kind === "model" ? "MODEL CARD" : "RECORDED REGRESSION CHECK";
  $("dialog-title").textContent =
    kind === "model" ? "MobileNetV3 Small" : "Evaluation results";
  if (!dialog.open) dialog.showModal();
  dialog.dataset.kind = kind;
  try {
    const data = await fetchJson(kind === "model" ? "/model" : "/evaluation", {
      signal: AbortSignal.timeout(5000),
    });
    if (dialog.dataset.kind !== kind) return;
    const content = $("dialog-content");
    content.replaceChildren();
    if (kind === "model") {
      content.append(
        paragraph(
          (data.weights === "Uploaded state dict"
            ? "Your uploaded MobileNetV3 Small weights. "
            : "A lightweight image classifier with pinned, pretrained ImageNet weights. ") +
            "The image is resized to 256 pixels, center-cropped to 224 × 224, and normalized using the standard ImageNet transforms.",
        ),
      );
      content.append(
        detailGrid([
          ["WEIGHTS", data.weights],
          ["COMPUTE", data.device.toUpperCase()],
          ["CLASSES", data.classes.toLocaleString()],
          ["PARAMETERS", data.parameters.toLocaleString()],
          ["INPUT", `${data.input_size.join(" × ")} pixels`],
          ["UPLOAD LIMIT", "10 MB / 20 megapixels"],
        ]),
      );
      const version = document.createElement("code");
      version.className = "dialog-code";
      version.textContent = `${data.model_version}\nSHA-256: ${data.weights_sha256}`;
      content.append(
        version,
        paragraph(
          "This model recognizes ImageNet categories. It may confidently misclassify images outside those categories, abstract images, or scenes with several objects. Scores are not calibrated certainty.",
        ),
      );
      const link = document.createElement("a");
      link.href =
        "https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html";
      link.target = "_blank";
      link.rel = "noopener";
      link.textContent = "Read the Torchvision model documentation ↗";
      content.append(link);
      content.append(
        paragraph(
          "Sample photography: Chelsea — Stefan van der Walt; coffee — Rachel Michetti; camera — Lav Varshney (CC0). The clock is public domain by Stefan van der Walt. Permissions are recorded in the repository’s evaluation manifest.",
        ),
      );
    } else {
      content.append(
        paragraph(
          `Recorded results for ${data.dataset}: ${data.sample_count} fixed, redistributable images. This report is bundled with the app; opening this panel does not rerun evaluation.`,
        ),
      );
      content.append(
        detailGrid([
          ["TOP-1 ACCURACY", `${(data.top1_accuracy * 100).toFixed(0)}%`],
          ["TOP-3 ACCURACY", `${(data.top3_accuracy * 100).toFixed(0)}%`],
          ["REQUIRED TOP-1", `${(data.min_top1_accuracy * 100).toFixed(0)}%`],
          ["GATE", data.passed ? "Passed" : "Failed"],
        ]),
      );
      const table = document.createElement("table");
      table.className = "evaluation-table";
      const head = table.createTHead().insertRow();
      ["SAMPLE", "TOP PREDICTION", "TOP-1"].forEach((text) => {
        const th = document.createElement("th");
        th.textContent = text;
        head.append(th);
      });
      const body = table.createTBody();
      data.results.forEach((sample) => {
        const row = body.insertRow();
        [
          sample.id,
          sample.predictions[0].label,
          sample.top1_correct ? "Pass" : "Miss",
        ].forEach((text) => {
          const cell = row.insertCell();
          cell.textContent = text;
        });
        if (!sample.top1_correct) row.className = "fail";
      });
      content.append(
        table,
        paragraph(
          "This four-image check detects obvious regressions. It is too small to estimate real-world accuracy. Human labels, accepted categories, image checksums, and redistribution permissions are fixed in the evaluation manifest.",
        ),
      );
      const version = document.createElement("code");
      version.className = "dialog-code";
      version.textContent = data.model_version;
      content.append(version);
    }
  } catch (error) {
    if (dialog.dataset.kind === kind)
      $("dialog-content").replaceChildren(paragraph(error.message));
  }
}

$("dropzone").addEventListener("click", () => $("file-input").click());
$("file-input").addEventListener("change", (event) =>
  selectFile(event.target.files[0]),
);
$("clear-image").addEventListener("click", resetSelection);
$("run-inference").addEventListener("click", runInference);
document
  .querySelectorAll("[data-sample]")
  .forEach((button) =>
    button.addEventListener("click", () => selectSample(button.dataset.sample)),
  );
document
  .querySelectorAll("[data-dialog]")
  .forEach((button) =>
    button.addEventListener("click", () => openDialog(button.dataset.dialog)),
  );
$("close-dialog").addEventListener("click", () => $("info-dialog").close());
$("info-dialog").addEventListener("click", (event) => {
  const rect = $("info-dialog").getBoundingClientRect();
  if (
    event.clientX < rect.left ||
    event.clientX > rect.right ||
    event.clientY < rect.top ||
    event.clientY > rect.bottom
  )
    $("info-dialog").close();
});
["dragenter", "dragover"].forEach((event) =>
  $("dropzone").addEventListener(event, (e) => {
    e.preventDefault();
    $("dropzone").classList.add("dragging");
  }),
);
["dragleave", "drop"].forEach((event) =>
  $("dropzone").addEventListener(event, (e) => {
    e.preventDefault();
    $("dropzone").classList.remove("dragging");
  }),
);
$("dropzone").addEventListener("drop", (event) =>
  selectFile(event.dataTransfer.files[0]),
);
// Prevent a dropped file outside the upload area from navigating away from the app.
window.addEventListener("dragover", (event) => event.preventDefault());
window.addEventListener("drop", (event) => event.preventDefault());
document.addEventListener("keydown", (event) => {
  if (
    event.key === "Enter" &&
    !$("playground-view").hidden &&
    !$("info-dialog").open &&
    !["BUTTON", "A", "INPUT"].includes(document.activeElement.tagName)
  ) {
    event.preventDefault();
    runInference();
  }
});
$("export-json").addEventListener("click", () => {
  if (!state.result) return;
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(state.result, null, 2)], {
      type: "application/json",
    }),
  );
  const link = document.createElement("a");
  link.href = url;
  link.download = `canary-${state.result.request_id}.json`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

checkService();
setInterval(checkService, 15_000);
fetchJson("/model")
  .then((model) => {
    $("app-version").textContent = model.app_version;
  })
  .catch(() => {});

const rollout = {
  data: null,
  busy: false,
  request: 0,
  checks: "",
  connected: false,
  polling: false,
};
const percent = (value) => `${Math.round(value * 100)}%`;

function modelLabel(version) {
  if (version?.includes("custom")) return "Your uploaded model";
  if (version?.includes("good")) return "Reliable release";
  if (version?.includes("bad")) return "Broken labels";
  return "Baseline";
}

function rolloutError(message) {
  $("rollout-error").textContent = message;
  $("rollout-error").hidden = !message;
}

function syncRolloutControls() {
  const active = Boolean(rollout.data?.candidate_version);
  const locked = rollout.busy || active;
  const upload = document.querySelector('[name="model-source"]:checked').value === "upload";
  const preset = document.querySelector('[name="preset"]:checked').value;
  const file = $("model-file").files[0];
  document.querySelectorAll('#model-source input, [name="preset"], #model-file')
    .forEach((input) => { input.disabled = locked; });
  $("model-options").hidden = upload;
  $("model-upload").hidden = !upload;
  $("model-file-name").textContent = file ? `${file.name} · ${(file.size / 1024 / 1024).toFixed(1)} MiB` : "No file selected";
  const selected = upload ? (file?.name || "Choose a model file") : preset === "good" ? "Reliable release" : "Broken labels";
  $("selected-model-name").textContent = active ? modelLabel(rollout.data.candidate_version) : selected;
  $("selected-model-description").textContent = active
    ? "Rollout in progress. You can return to the playground while the checks run."
    : upload
      ? "Compatible weights will be checked before receiving more traffic."
      : preset === "good"
        ? "Expected to pass all three checks and become your current model."
        : "Expected to fail the first check and automatically restore your current model.";
  $("start-rollout").disabled = locked || !rollout.connected || (upload && !file);
  $("start-rollout").textContent = rollout.busy ? "Starting rollout…"
    : active ? "Rollout in progress…"
      : upload ? "Roll out uploaded model →"
        : preset === "good" ? "Roll out reliable release →" : "Roll out broken labels →";
  $("rollback-model").hidden = !active;
  $("advance-model").hidden = !active || rollout.data?.automatic;
  const checking = rollout.data?.status === "evaluating";
  $("rollback-model").disabled = rollout.busy || !rollout.connected || checking;
  $("advance-model").disabled = rollout.busy || !rollout.connected || checking;
}

function renderRolloutChecks(checks) {
  const signature = JSON.stringify(checks);
  if (rollout.checks === signature) return;
  rollout.checks = signature;
  const container = $("rollout-checks");
  container.replaceChildren();
  if (!checks.length) return;
  const table = document.createElement("table");
  table.className = "gate-table";
  table.createCaption().textContent = "Accuracy checks · fixed sample dataset";
  const head = table.createTHead().insertRow();
  ["Traffic", "Top-1", "Top-3", "Result"].forEach((label) => {
    const cell = document.createElement("th");
    cell.scope = "col";
    cell.textContent = label;
    head.append(cell);
  });
  const body = table.createTBody();
  checks.forEach((check) => {
    const row = body.insertRow();
    row.className = check.passed ? "gate-pass" : "gate-fail";
    [
      `${check.candidate_percent}%`,
      percent(check.top1_accuracy),
      percent(check.top3_accuracy),
      check.passed ? "Passed" : "Failed → rollback",
    ].forEach((value) => { row.insertCell().textContent = value; });
  });
  const details = document.createElement("details");
  const summary = document.createElement("summary");
  summary.textContent = "See what the candidate predicted";
  details.append(summary);
  checks.at(-1).results.forEach((sample) => {
    details.append(paragraph(
      `${sample.id}: ${sample.predictions[0].label} — expected ${sample.expected_labels.join(" or ")} (${sample.top1_correct ? "correct" : "miss"})`,
    ));
  });
  container.append(table, details);
}

function renderRolloutPanel(data) {
  // Reopening or refreshing a running rollout should show its actual source.
  if (data.candidate_version && data.candidate_version !== rollout.data?.candidate_version) {
    const custom = data.candidate_version.includes("custom");
    document.querySelector(`[name="model-source"][value="${custom ? "upload" : "preset"}"]`).checked = true;
    if (!custom) {
      const preset = data.candidate_version.includes("bad") ? "bad" : "good";
      document.querySelector(`[name="preset"][value="${preset}"]`).checked = true;
    }
  }
  rollout.data = data;
  const active = Boolean(data.candidate_version);
  const failed = data.status === "rolled_back";
  const promoted = data.status === "promoted";
  const traffic = data.candidate_percent;
  const names = {
    idle: "Stable", canary: "Rolling out", evaluating: "Checking",
    promoted: "Promoted", rolled_back: "Rolled back",
  };
  $("rollout-badge").textContent = names[data.status] || data.status;
  $("rollout-panel").dataset.status = data.status;
  $("current-model-name").textContent = modelLabel(data.stable_version);
  $("current-model-version").textContent = data.stable_version;
  const caption = `Stable ${100 - traffic}% · Candidate ${traffic}%`;
  $("traffic-caption").textContent = caption;
  $("traffic-track").setAttribute("aria-label", caption);
  $("stable-traffic").style.width = `${100 - traffic}%`;
  $("candidate-traffic").style.width = `${traffic}%`;
  document.querySelectorAll("[data-stage]").forEach((stage) => {
    const check = data.checks.find((item) => item.candidate_percent === Number(stage.dataset.stage));
    const current = active && traffic === Number(stage.dataset.stage);
    stage.dataset.state = check ? (check.passed ? "passed" : "failed") : current ? "active" : "waiting";
    stage.querySelector("small").textContent = check
      ? (check.passed ? "✓ Passed" : "× Failed")
      : current ? (data.automatic ? "Checking…" : "Awaiting check")
        : failed ? "Skipped" : "Waiting";
    if (current) stage.setAttribute("aria-current", "step");
    else stage.removeAttribute("aria-current");
  });
  const finalStage = $("rollout-final-stage");
  finalStage.dataset.state = promoted ? "passed" : failed ? "failed" : "waiting";
  finalStage.querySelector("span").textContent = failed ? "Restore stable" : "Promote model";
  finalStage.querySelector("small").textContent = promoted ? "✓ Live" : failed ? "↶ Restored" : "Waiting";
  let title = "Ready when you are";
  let detail = "Choose a model and start a rollout. Progress and check results will appear here.";
  if (active) {
    title = `${traffic}% of traffic goes to ${modelLabel(data.candidate_version).toLowerCase()}`;
    detail = data.automatic
      ? "Checks run automatically. If accuracy drops below the gate, traffic returns to your current model."
      : "Use Check and advance to evaluate this manually started rollout.";
  } else if (promoted) {
    title = "Rollout successful — new model is live";
    detail = `${modelLabel(data.stable_version)} passed all three checks and now serves 100% of traffic.`;
  } else if (failed) {
    title = data.reason?.startsWith("Manual")
      ? "Rollout stopped — stable model restored"
      : "Automatic rollback — stable model restored";
    detail = `${modelLabel(data.last_candidate_version)} was withdrawn. ${modelLabel(data.stable_version)} serves 100% of traffic. ${data.reason}`;
  }
  if ($("rollout-result-title").textContent !== title) $("rollout-result-title").textContent = title;
  if ($("rollout-result-detail").textContent !== detail) $("rollout-result-detail").textContent = detail;
  renderRolloutChecks(data.checks);
  syncRolloutControls();
}

async function refreshRollout() {
  // Keep one poll in flight; slow responses must not invalidate each other.
  if (rollout.busy || rollout.polling) return;
  rollout.polling = true;
  const request = rollout.request;
  try {
    const data = await fetchJson("/rollout", {
      cache: "no-store", signal: AbortSignal.timeout(5000),
    });
    if (request !== rollout.request) return;
    rollout.connected = true;
    $("rollout-connection").hidden = true;
    renderRolloutPanel(data);
  } catch {
    if (request !== rollout.request) return;
    rollout.connected = false;
    $("rollout-badge").textContent = "Offline";
    $("rollout-connection").textContent = "Cannot reach the rollout service. You can choose a model while we reconnect; starting a rollout will be available when the connection returns.";
    $("rollout-connection").hidden = false;
    syncRolloutControls();
  } finally {
    rollout.polling = false;
  }
}

async function changeRollout(path, body) {
  if (rollout.busy) return;
  rollout.busy = true;
  ++rollout.request;
  rolloutError("");
  syncRolloutControls();
  try {
    const data = await fetchJson(path, { method: "POST", ...(body ? { body } : {}) });
    rollout.connected = true;
    $("rollout-connection").hidden = true;
    renderRolloutPanel(data);
  } catch (error) {
    rolloutError(error.message);
  } finally {
    rollout.busy = false;
    syncRolloutControls();
    await refreshRollout();
  }
}

$("start-rollout").addEventListener("click", () => {
  const upload = document.querySelector('[name="model-source"]:checked').value === "upload";
  if (upload) {
    const file = $("model-file").files[0];
    if (!file || !file.size) return rolloutError("Choose a non-empty model file.");
    if (file.size > 32 * 1024 * 1024) return rolloutError("Model files must be 32 MiB or smaller.");
    const body = new FormData();
    body.append("file", file);
    changeRollout("/rollout/upload", body);
  } else {
    const preset = document.querySelector('[name="preset"]:checked').value;
    changeRollout(`/rollout/start/${preset}`);
  }
});
$("rollback-model").addEventListener("click", () => changeRollout("/rollout/rollback"));
$("advance-model").addEventListener("click", () => changeRollout("/rollout/advance"));
document.querySelectorAll('[name="model-source"], [name="preset"], #model-file')
  .forEach((input) => input.addEventListener("change", () => {
    rolloutError("");
    syncRolloutControls();
  }));

function showWorkspace(focus = false) {
  const isRollout = location.hash === "#rollout";
  $("playground-view").hidden = isRollout;
  $("rollout-view").hidden = !isRollout;
  $("current-page").textContent = isRollout ? "Canary rollout" : "Playground";
  document.title = `CanaryVision · ${isRollout ? "Canary rollout" : "Inference playground"}`;
  [$("show-playground"), $("show-rollout")].forEach((link) => {
    const selected = link.id === (isRollout ? "show-rollout" : "show-playground");
    link.classList.toggle("active", selected);
    if (selected) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
  if (focus) {
    window.scrollTo({ top: 0, behavior: "instant" });
    $(isRollout ? "rollout-title" : "page-title").focus({ preventScroll: true });
  }
  if (isRollout) refreshRollout();
}
window.addEventListener("hashchange", () => showWorkspace(true));
showWorkspace();
syncRolloutControls();
refreshRollout();
setInterval(() => {
  if (!$("rollout-view").hidden || rollout.data?.candidate_version) refreshRollout();
}, 1000);
