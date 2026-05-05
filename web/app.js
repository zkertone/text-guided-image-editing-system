const DEFAULT_API_BASE = "http://localhost:8000";
let apiBase = normalizeApiBase(
  localStorage.getItem("backendApiBase") || window.API_BASE || DEFAULT_API_BASE
);

let loadedImageBlob = null;
let loadedImageName = "history_input.png";

const elements = {
  serverStatus: document.getElementById("serverStatus"),
  apiBaseInput: document.getElementById("apiBaseInput"),
  saveApiBaseButton: document.getElementById("saveApiBaseButton"),
  testApiBaseButton: document.getElementById("testApiBaseButton"),
  editForm: document.getElementById("editForm"),
  mode: document.getElementById("mode"),
  imageInput: document.getElementById("imageInput"),
  maskInput: document.getElementById("maskInput"),
  maskField: document.getElementById("maskField"),
  prompt: document.getElementById("prompt"),
  steps: document.getElementById("steps"),
  imageGuidance: document.getElementById("imageGuidance"),
  imageGuidanceField: document.getElementById("imageGuidanceField"),
  guidance: document.getElementById("guidance"),
  outputImage: document.getElementById("outputImage"),
  controlImage: document.getElementById("controlImage"),
  controlBlock: document.getElementById("controlBlock"),
  summaryText: document.getElementById("summaryText"),
  historyBody: document.getElementById("historyBody"),
  inputImagesBody: document.getElementById("inputImagesBody"),
  recordIdInput: document.getElementById("recordIdInput"),
  viewRecordButton: document.getElementById("viewRecordButton"),
  recordDetail: document.getElementById("recordDetail"),
  historyInputImage: document.getElementById("historyInputImage"),
  historyMaskImage: document.getElementById("historyMaskImage"),
  historyControlImage: document.getElementById("historyControlImage"),
  historyOutputImage: document.getElementById("historyOutputImage"),
  loadImageIdInput: document.getElementById("loadImageIdInput"),
  loadInputButton: document.getElementById("loadInputButton"),
  loadedImageHint: document.getElementById("loadedImageHint"),
  currentInputPreview: document.getElementById("currentInputPreview"),
  deleteRecordIdInput: document.getElementById("deleteRecordIdInput"),
  deleteRecordButton: document.getElementById("deleteRecordButton"),
  deleteImageIdInput: document.getElementById("deleteImageIdInput"),
  deleteImageButton: document.getElementById("deleteImageButton"),
  deleteStatus: document.getElementById("deleteStatus"),
};

function normalizeApiBase(value) {
  const trimmed = (value || DEFAULT_API_BASE).trim();
  return trimmed.endsWith("/") ? trimmed.slice(0, -1) : trimmed;
}

function getApiBase() {
  return apiBase;
}

function apiUrl(path) {
  return `${getApiBase()}${path}`;
}

function setImage(img, url) {
  if (!url) {
    img.removeAttribute("src");
    return;
  }
  img.src = apiUrl(url);
}

function updateModeControls() {
  const mode = elements.mode.value;
  elements.maskField.classList.toggle("hidden", mode !== "local_inpaint");
  elements.imageGuidanceField.classList.toggle("hidden", mode !== "global_edit");
  elements.controlBlock.classList.toggle("hidden", mode !== "controlnet_canny");
}

async function fetchJson(path, options = {}) {
  const response = await fetch(apiUrl(path), options);
  const contentType = response.headers.get("content-type") || "";
  const body = contentType.includes("application/json")
    ? await response.json()
    : await response.text();

  if (!response.ok) {
    const message = typeof body === "string" ? body : body.detail || "请求失败";
    throw new Error(message);
  }

  return body;
}

async function checkHealth() {
  try {
    const data = await fetchJson("/api/health");
    elements.serverStatus.textContent = `连接状态：后端连接成功 (${data.status})`;
  } catch (error) {
    elements.serverStatus.textContent = "连接状态：后端连接失败，请检查地址或后端服务";
  }
}

function saveApiBase() {
  apiBase = normalizeApiBase(elements.apiBaseInput.value);
  elements.apiBaseInput.value = apiBase;
  localStorage.setItem("backendApiBase", apiBase);
  elements.serverStatus.textContent = "连接状态：后端地址已保存";
}

async function testApiBase() {
  apiBase = normalizeApiBase(elements.apiBaseInput.value);
  elements.apiBaseInput.value = apiBase;
  try {
    const response = await fetch(`${apiBase}/api/health`);
    if (!response.ok) {
      throw new Error("health check failed");
    }
    localStorage.setItem("backendApiBase", apiBase);
    elements.serverStatus.textContent = "连接状态：后端连接成功";
  } catch (error) {
    elements.serverStatus.textContent = "连接状态：后端连接失败，请检查地址或后端服务";
  }
}

function renderHistory(records) {
  elements.historyBody.innerHTML = "";
  records.forEach((record) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${record.id}</td>
      <td>${record.created_at}</td>
      <td>${record.mode}</td>
      <td>${record.prompt}</td>
      <td>${record.status}</td>
    `;
    tr.addEventListener("click", () => {
      elements.recordIdInput.value = record.id;
      viewRecord(record.id);
    });
    elements.historyBody.appendChild(tr);
  });
}

function renderInputImages(images) {
  elements.inputImagesBody.innerHTML = "";
  images.forEach((image) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${image.id}</td>
      <td>${image.file_name}</td>
      <td>${image.created_at}</td>
      <td>${image.width}</td>
      <td>${image.height}</td>
    `;
    tr.addEventListener("click", () => {
      elements.loadImageIdInput.value = image.id;
    });
    elements.inputImagesBody.appendChild(tr);
  });
}

async function refreshHistory() {
  const records = await fetchJson("/api/history");
  renderHistory(records);
}

async function refreshInputImages() {
  const images = await fetchJson("/api/input-images");
  renderInputImages(images);
}

async function submitEdit(event) {
  event.preventDefault();

  const formData = new FormData();
  const mode = elements.mode.value;
  const selectedFile = elements.imageInput.files[0];

  if (selectedFile) {
    formData.append("image", selectedFile);
  } else if (loadedImageBlob) {
    formData.append("image", loadedImageBlob, loadedImageName);
  } else {
    elements.summaryText.textContent = "请先上传输入图像，或加载一张历史输入图。";
    return;
  }

  if (mode === "local_inpaint") {
    const maskFile = elements.maskInput.files[0];
    if (!maskFile) {
      elements.summaryText.textContent = "局部编辑模式下，请上传 Mask 图。";
      return;
    }
    formData.append("mask_image", maskFile);
  }

  formData.append("mode", mode);
  formData.append("prompt", elements.prompt.value);
  formData.append("num_inference_steps", elements.steps.value);
  formData.append("image_guidance_scale", elements.imageGuidance.value);
  formData.append("guidance_scale", elements.guidance.value);

  elements.summaryText.textContent = "正在编辑，请等待模型推理完成...";

  try {
    const result = await fetchJson("/api/edit", {
      method: "POST",
      body: formData,
    });

    setImage(elements.outputImage, result.output_image_url);
    setImage(elements.controlImage, result.control_image_url);
    elements.controlBlock.classList.toggle("hidden", !result.control_image_url);
    elements.summaryText.textContent = result.summary_text || "";

    await refreshHistory();
    await refreshInputImages();
  } catch (error) {
    elements.summaryText.textContent = `编辑失败: ${error.message}`;
  }
}

async function viewRecord(recordId) {
  const id = recordId || elements.recordIdInput.value;
  if (!id) {
    elements.recordDetail.textContent = "请输入 record_id。";
    return;
  }

  try {
    const detail = await fetchJson(`/api/history/${id}`);
    const images = await fetchJson(`/api/history/${id}/images`);
    elements.recordDetail.textContent = JSON.stringify(detail, null, 2);
    setImage(elements.historyInputImage, images.input_image_url);
    setImage(elements.historyMaskImage, images.mask_image_url);
    setImage(elements.historyControlImage, images.control_image_url);
    setImage(elements.historyOutputImage, images.output_image_url);
  } catch (error) {
    elements.recordDetail.textContent = `查看记录失败: ${error.message}`;
  }
}

async function loadInputImage() {
  const imageId = elements.loadImageIdInput.value;
  if (!imageId) {
    elements.loadedImageHint.textContent = "请输入 image_id。";
    return;
  }

  try {
    const response = await fetch(apiUrl(`/api/images/${imageId}`));
    if (!response.ok) {
      throw new Error("图片不存在或已删除。");
    }
    loadedImageBlob = await response.blob();
    loadedImageName = `history_input_${imageId}.png`;
    elements.currentInputPreview.src = URL.createObjectURL(loadedImageBlob);
    elements.loadedImageHint.textContent = `已加载历史输入图，image_id=${imageId}`;
  } catch (error) {
    elements.loadedImageHint.textContent = `加载失败: ${error.message}`;
  }
}

async function deleteRecord() {
  const recordId = elements.deleteRecordIdInput.value;
  if (!recordId) {
    elements.deleteStatus.textContent = "请输入 record_id。";
    return;
  }

  try {
    await fetchJson(`/api/history/${recordId}`, { method: "DELETE" });
    elements.deleteStatus.textContent = `已逻辑删除记录，record_id=${recordId}`;
    await refreshHistory();
  } catch (error) {
    elements.deleteStatus.textContent = `删除记录失败: ${error.message}`;
  }
}

async function deleteImage() {
  const imageId = elements.deleteImageIdInput.value;
  if (!imageId) {
    elements.deleteStatus.textContent = "请输入 image_id。";
    return;
  }

  try {
    await fetchJson(`/api/images/${imageId}`, { method: "DELETE" });
    elements.deleteStatus.textContent = `已逻辑删除图片，image_id=${imageId}`;
    await refreshHistory();
    await refreshInputImages();
  } catch (error) {
    elements.deleteStatus.textContent = `删除图片失败: ${error.message}`;
  }
}

elements.mode.addEventListener("change", updateModeControls);
elements.saveApiBaseButton.addEventListener("click", saveApiBase);
elements.testApiBaseButton.addEventListener("click", testApiBase);
elements.imageInput.addEventListener("change", () => {
  const file = elements.imageInput.files[0];
  loadedImageBlob = null;
  if (file) {
    elements.currentInputPreview.src = URL.createObjectURL(file);
    elements.loadedImageHint.textContent = "";
  }
});
elements.editForm.addEventListener("submit", submitEdit);
elements.viewRecordButton.addEventListener("click", () => viewRecord());
elements.loadInputButton.addEventListener("click", loadInputImage);
elements.deleteRecordButton.addEventListener("click", deleteRecord);
elements.deleteImageButton.addEventListener("click", deleteImage);

elements.apiBaseInput.value = apiBase;
updateModeControls();
checkHealth();
refreshHistory();
refreshInputImages();
