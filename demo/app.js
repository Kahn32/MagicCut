const state = { meshes: [], result: null };
const $ = (id) => document.getElementById(id);

async function request(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || `${response.status} ${response.statusText}`);
  }
  return response.json();
}

function selectedMesh() {
  return state.meshes.find((item) => item.uid === $("mesh").value);
}

function populateQueries() {
  const mesh = selectedMesh();
  $("query").replaceChildren(...mesh.part_ids.map((id) => new Option(String(id), String(id))));
  $("message").textContent = `${mesh.part_count} representative parts. Choose a query and run grouping.`;
  renderParts(null);
}

function partClass(id, result) {
  if (!result) return "negative";
  if (id === result.query_part) return "query";
  if (result.uncertain.includes(id)) return "uncertain";
  return result.selected.includes(id) ? "positive" : "negative";
}

function renderParts(result) {
  const mesh = selectedMesh();
  const shown = mesh.part_ids.slice(0, 500);
  const cards = shown.map((id) => {
    const card = document.createElement("button");
    card.className = `part ${partClass(id, result)}`;
    if (result?.recommended_part === id) card.classList.add("recommended");
    card.dataset.part = id;
    card.title = `Part ${id}`;
    const image = document.createElement("img");
    image.loading = "lazy";
    image.alt = `Rendered part ${id}`;
    image.src = `/api/render/${mesh.uid}/${id}`;
    const label = document.createElement("span");
    label.textContent = id;
    card.append(image, label);
    card.addEventListener("click", () => { $("query").value = String(id); });
    return card;
  });
  $("parts").replaceChildren(...cards);
  if (mesh.part_ids.length > shown.length) {
    $("message").textContent += ` Showing the first ${shown.length} parts for browser responsiveness.`;
  }
}

function update(result) {
  state.result = result;
  $("clicks").textContent = result.click_count;
  $("latency").textContent = `${result.inference_ms.toFixed(1)} ms`;
  $("recommendation").textContent = result.recommended_part ?? "Complete";
  const p = result.recommended_part == null ? null : result.probabilities[String(result.recommended_part)];
  $("confidence").textContent = p == null ? "—" : `${(100 * Math.max(p, 1 - p)).toFixed(1)}%`;
  $("feedback").hidden = result.recommended_part == null;
  $("reset").disabled = false;
  $("message").textContent = `${result.selected.length} selected; ${result.uncertain.length} uncertain.`;
  renderParts(result);
}

async function run() {
  $("message").textContent = "Running frozen baseline and MagicCut…";
  try {
    update(await request("/api/start", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({uid: $("mesh").value, query_part: Number($("query").value)})
    }));
  } catch (error) { $("message").textContent = error.message; }
}

async function answer(positive) {
  const result = state.result;
  if (!result?.recommended_part) return;
  try {
    update(await request("/api/feedback", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({session_id: result.session_id, part_id: result.recommended_part, positive})
    }));
  } catch (error) { $("message").textContent = error.message; }
}

async function reset() {
  if (!state.result) return;
  try {
    update(await request("/api/reset", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({session_id: state.result.session_id})
    }));
  } catch (error) { $("message").textContent = error.message; }
}

async function initialize() {
  try {
    state.meshes = await request("/api/meshes");
    $("mesh").replaceChildren(...state.meshes.map((mesh) => new Option(`${mesh.uid.slice(0, 8)} — ${mesh.part_count} parts`, mesh.uid)));
    populateQueries();
  } catch (error) { $("message").textContent = error.message; }
}

$("mesh").addEventListener("change", populateQueries);
$("run").addEventListener("click", run);
$("yes").addEventListener("click", () => answer(true));
$("no").addEventListener("click", () => answer(false));
$("reset").addEventListener("click", reset);
initialize();
