(() => {
  "use strict";

  const { concepts, categories, practices, glossary, tomographyViews, companyTypeNames } = window.ATLAS_CONTENT;
  const byId = (id) => document.getElementById(id);
  const conceptById = new Map(concepts.map((concept) => [concept.id, concept]));
  const specialTopicLabels = { tomografia: "Tomografía · entrenamiento y predicción" };
  const STORE_NOTES = "atlas-multimodal-notes-v1";
  const STORE_PROGRESS = "atlas-multimodal-progress-v1";
  const LAB_URL = "http://127.0.0.1:8501/";
  const LAB_MODULES = {
    inicio: { title: "Laboratorio multimodal", description: "Explora las pruebas de texto e imagen en Streamlit local.", query: "" },
    aritmetica: { title: "Vectores A, B y C", description: "Compara ángulos reales y prueba A − B + C en el espacio de embeddings.", query: "aritmetica" },
    tomografia: { title: "Tomografía de representaciones", description: "Inspecciona capas, mapas de embeddings y la sonda entrenada en el laboratorio.", query: "tomografia" },
    caso_empresa: { title: "Caso empresarial", description: "Clasifica páginas públicas y examina sus vecinos en el espacio 3D.", query: "caso_empresa" },
    comparador: { title: "Comparador multimodal", description: "Carga imágenes, añade texto y compara posiciones y similitudes.", query: "comparador" }
  };
  const ATLAS_SECTIONS = new Set(["inicio", "mapa", "catalogo", "explorador", "tomografia", "casos", "practicas", "bitacora", "glosario"]);
  let notes = readStored(STORE_NOTES, []);
  let completed = readStored(STORE_PROGRESS, {});
  let selectedCategory = "todos";
  let companyDocs = [];
  let toastTimer;
  let lastAtlasHash = "#inicio";

  function readStored(key, fallback) {
    try {
      const value = JSON.parse(localStorage.getItem(key));
      return value && typeof value === "object" ? value : fallback;
    } catch {
      return fallback;
    }
  }

  if (!Array.isArray(notes)) notes = [];
  if (!completed || Array.isArray(completed)) completed = {};

  function saveStored(key, value) {
    try {
      localStorage.setItem(key, JSON.stringify(value));
      return true;
    } catch {
      showToast("No se pudo guardar en este navegador. Exporta tus notas para conservarlas.");
      return false;
    }
  }

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (char) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
    })[char]);
  }

  function fold(value) {
    return String(value).normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase();
  }

  function showToast(message) {
    const toast = byId("toast");
    toast.textContent = message;
    toast.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.remove("show"), 3500);
  }

  function topicLabel(id) {
    return conceptById.get(id)?.title || specialTopicLabels[id] || "General";
  }

  function renderFilters() {
    byId("conceptFilters").innerHTML = categories.map((category) =>
      `<button type="button" class="filter-button ${category.id === selectedCategory ? "active" : ""}" data-filter="${category.id}" aria-pressed="${category.id === selectedCategory}">${escapeHtml(category.label)}</button>`
    ).join("");
    byId("conceptFilters").addEventListener("click", (event) => {
      const button = event.target.closest("[data-filter]");
      if (!button) return;
      selectedCategory = button.dataset.filter;
      renderFiltersState();
      renderConcepts();
    });
  }

  function renderFiltersState() {
    for (const button of byId("conceptFilters").querySelectorAll("[data-filter]")) {
      const active = button.dataset.filter === selectedCategory;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
    }
  }

  function renderConcepts() {
    const query = fold(byId("conceptSearch").value.trim());
    const visible = concepts.filter((concept) =>
      (selectedCategory === "todos" || concept.category === selectedCategory) &&
      fold([concept.title, concept.short, concept.summary, concept.step].join(" ")).includes(query)
    );
    byId("conceptGrid").innerHTML = visible.map((concept) =>
      `<button type="button" class="concept-card" data-concept="${concept.id}" aria-label="Abrir concepto: ${escapeHtml(concept.title)}">
        <span class="concept-top"><span class="concept-icon icon-${concept.category}" aria-hidden="true">${escapeHtml(concept.icon)}</span><span class="concept-step">${escapeHtml(concept.step)}</span></span>
        <h3>${escapeHtml(concept.title)}</h3><p>${escapeHtml(concept.short)}</p><span class="concept-footer">Abrir ficha <span aria-hidden="true">→</span></span>
      </button>`
    ).join("");
    byId("conceptEmpty").hidden = visible.length !== 0;
    byId("conceptCount").textContent = `${visible.length} ${visible.length === 1 ? "FICHA" : "FICHAS"}`;
  }

  function openConcept(id) {
    const concept = conceptById.get(id);
    if (!concept) return;
    byId("dialogContent").innerHTML = `
      <p class="dialog-kicker">${escapeHtml(concept.step)} · ${escapeHtml(categories.find((item) => item.id === concept.category)?.label || "Concepto")}</p>
      <h2>${escapeHtml(concept.title)}</h2>
      <p class="dialog-summary">${escapeHtml(concept.summary)}</p>
      <div class="dialog-example"><strong>Ejemplo:</strong> ${escapeHtml(concept.example)}</div>
      <div class="dialog-section"><h3>Por qué importa</h3><p>${escapeHtml(concept.why)}</p></div>
      <div class="dialog-section"><h3>Prueba en el laboratorio</h3><p>${escapeHtml(concept.test)}</p></div>
      <div class="dialog-section"><h3>Al interpretar</h3><p>${escapeHtml(concept.caution)}</p></div>
      <div class="dialog-actions"><button type="button" class="button button-primary" data-note-concept="${concept.id}">Anotar una idea →</button><a class="button button-outline" href="#lab/inicio">Abrir laboratorio →</a></div>`;
    const dialog = byId("conceptDialog");
    if (!dialog.open) dialog.showModal();
  }

  function renderPractices() {
    byId("practiceGrid").innerHTML = practices.map((practice) => `
      <article class="practice-card">
        <div class="practice-top"><span class="practice-num">${practice.number}</span><div><h3>${escapeHtml(practice.title)}</h3><p>${escapeHtml(practice.subtitle)}</p></div></div>
        <ol class="practice-steps">${practice.steps.map((step) => `<li>${escapeHtml(step)}</li>`).join("")}</ol>
        <p class="practice-question"><strong>Pregunta guía:</strong> ${escapeHtml(practice.question)}</p>
        <div class="practice-actions"><button type="button" class="button button-quiet button-small" data-note-concept="${practice.concept}">Anotar resultado ✎</button><label class="done-label"><input type="checkbox" data-practice="${practice.id}" ${completed[practice.id] ? "checked" : ""}> Hecha</label></div>
      </article>`).join("");
    updateProgress();
  }

  function updateProgress() {
    const count = practices.filter((practice) => completed[practice.id]).length;
    byId("progressText").textContent = `${count}/${practices.length}`;
    byId("progressFill").style.width = `${count / practices.length * 100}%`;
  }

  function renderGlossary() {
    byId("glossaryGrid").innerHTML = glossary.map((entry) =>
      `<div class="glossary-item"><strong>${escapeHtml(entry.term)}</strong><p>${escapeHtml(entry.definition)}</p></div>`
    ).join("");
  }

  function updateCosine() {
    const degrees = Number(byId("angleSlider").value);
    const length = Number(byId("lengthSlider").value);
    const radians = degrees * Math.PI / 180;
    const x = 180 + 94 * length * Math.cos(radians);
    const y = 130 - 94 * length * Math.sin(radians);
    const cosine = Math.cos(radians);
    const rawDot = length * cosine;
    const vectorB = byId("vectorB");
    vectorB.setAttribute("x2", String(x));
    vectorB.setAttribute("y2", String(y));
    byId("vectorBLabel").setAttribute("x", String(Math.max(12, Math.min(335, x + (cosine >= 0 ? 8 : -19)))));
    byId("vectorBLabel").setAttribute("y", String(Math.max(17, Math.min(247, y - 8))));
    byId("angleValue").textContent = `${degrees}°`;
    byId("lengthValue").textContent = `${length.toFixed(1)}×`;
    byId("cosineValue").textContent = cosine.toFixed(3).replace(".", ",");
    const label = cosine > 0.75 ? "dirección cercana" : cosine > 0.2 ? "parecido moderado" : cosine > -0.2 ? "casi perpendicular" : "dirección opuesta";
    byId("cosineLabel").textContent = label;
    byId("cosineExplanation").textContent = `Producto punto en este dibujo: ${rawDot.toFixed(2).replace(".", ",")}. El coseno depende del ángulo: cambiar solo la longitud de B no altera su valor.`;
  }

  const triCamera = { yaw: 35, pitch: 28 };
  const triPresets = {
    separados: { A: [20, 15], B: [100, 35], C: [-125, 25] },
    cercanos: { A: [15, 20], B: [28, 18], C: [42, 22] },
    opuestos: { A: [0, 0], B: [180, 0], C: [90, 45] }
  };
  const triFormat = (value, places = 3) => (Math.abs(value) < 0.5 * 10 ** -places ? 0 : value)
    .toFixed(places).replace(".", ",").replace("-", "−");

  function updateTriVectors() {
    const cx = 240, cy = 185, scale = 120;
    const radians = Math.PI / 180;
    const yaw = triCamera.yaw * radians, pitch = triCamera.pitch * radians;
    const project = ({ x, y, z }, radius = scale) => {
      const rotatedX = x * Math.cos(yaw) - y * Math.sin(yaw);
      const rotatedY = x * Math.sin(yaw) + y * Math.cos(yaw);
      return { x: cx + radius * rotatedX, y: cy - radius * (z * Math.cos(pitch) + rotatedY * Math.sin(pitch)) };
    };
    const unit = (azimuth, elevation) => ({
      x: Math.cos(elevation * radians) * Math.cos(azimuth * radians),
      y: Math.cos(elevation * radians) * Math.sin(azimuth * radians),
      z: Math.sin(elevation * radians)
    });
    const dot = (a, b) => a.x * b.x + a.y * b.y + a.z * b.z;
    const cross = (a, b) => ({ x: a.y * b.z - a.z * b.y, y: a.z * b.x - a.x * b.z, z: a.x * b.y - a.y * b.x });
    const normalize = (v) => {
      const size = Math.hypot(v.x, v.y, v.z) || 1;
      return { x: v.x / size, y: v.y / size, z: v.z / size };
    };
    const setLine = (id, a, b) => {
      const line = byId(id);
      line.setAttribute("x1", a.x.toFixed(2));
      line.setAttribute("y1", a.y.toFixed(2));
      line.setAttribute("x2", b.x.toFixed(2));
      line.setAttribute("y2", b.y.toFixed(2));
    };
    const origin = { x: cx, y: cy };
    for (const [axis, direction] of [["X", { x: 1, y: 0, z: 0 }], ["Y", { x: 0, y: 1, z: 0 }], ["Z", { x: 0, y: 0, z: 1 }]]) {
      const end = project(direction, 145);
      setLine(`triAxis${axis}`, origin, end);
      byId(`triAxisLabel${axis}`).setAttribute("x", String(Math.max(12, Math.min(458, end.x + 7))));
      byId(`triAxisLabel${axis}`).setAttribute("y", String(Math.max(20, Math.min(328, end.y - 7))));
    }
    byId("triGround").setAttribute("cy", String(cy));
    byId("triGround").setAttribute("ry", String(Math.max(4, 120 * Math.abs(Math.sin(pitch)))));
    const vectors = {}, tips = {};
    for (const name of ["A", "B", "C"]) {
      const azimuth = Number(byId(`triSlider${name}`).value);
      const elevation = Number(byId(`triElev${name}`).value);
      byId(`triValue${name}`).textContent = `${azimuth}°`.replace("-", "−");
      byId(`triElevValue${name}`).textContent = `${elevation}°`.replace("-", "−");
      vectors[name] = unit(azimuth, elevation);
      tips[name] = project(vectors[name]);
      setLine(`triVector${name}`, origin, tips[name]);
      const label = byId(`triLabel${name}`);
      const rightSide = tips[name].x >= cx;
      label.setAttribute("x", String(Math.max(22, Math.min(458, tips[name].x + (rightSide ? 10 : -10)))));
      label.setAttribute("y", String(Math.max(22, Math.min(325, tips[name].y - 8))));
      label.setAttribute("text-anchor", rightSide ? "start" : "end");
    }
    const pairs = [["A", "B", 40], ["B", "C", 57], ["A", "C", 74]];
    const summaries = [];
    for (const [first, second, radius] of pairs) {
      const key = first + second;
      const a = vectors[first], b = vectors[second];
      const cosine = Math.max(-1, Math.min(1, dot(a, b)));
      const theta = Math.acos(cosine);
      setLine(`triDiff${key}`, tips[first], tips[second]);
      let arcPath = "";
      if (theta > 1e-4) {
        let perpendicular = normalize(cross(a, { x: 0, y: 0, z: 1 }));
        if (Math.abs(dot(a, { x: 0, y: 0, z: 1 })) > 0.97) perpendicular = normalize(cross(a, { x: 0, y: 1, z: 0 }));
        const samples = 28;
        const points = [];
        for (let i = 0; i <= samples; i++) {
          const t = i / samples;
          let v;
          if (Math.abs(Math.sin(theta)) < 1e-4) {
            v = { x: Math.cos(t * theta) * a.x + Math.sin(t * theta) * perpendicular.x,
              y: Math.cos(t * theta) * a.y + Math.sin(t * theta) * perpendicular.y,
              z: Math.cos(t * theta) * a.z + Math.sin(t * theta) * perpendicular.z };
          } else {
            const wa = Math.sin((1 - t) * theta) / Math.sin(theta);
            const wb = Math.sin(t * theta) / Math.sin(theta);
            v = { x: wa * a.x + wb * b.x, y: wa * a.y + wb * b.y, z: wa * a.z + wb * b.z };
          }
          points.push(project(v, radius));
        }
        arcPath = points.map((point, index) => `${index ? "L" : "M"}${point.x.toFixed(2)} ${point.y.toFixed(2)}`).join(" ");
      }
      byId(`triArc${key}`).setAttribute("d", arcPath);
      const degrees = theta / radians;
      byId(`triAngle${key}`).textContent = `${triFormat(degrees, 1)}°`;
      byId(`triCos${key}`).textContent = triFormat(cosine);
      byId(`triDistance${key}`).textContent = triFormat(Math.sqrt(Math.max(0, 2 - 2 * cosine)));
      summaries.push(`${key} ${triFormat(degrees, 1)} grados`);
    }
    byId("triSvg").setAttribute("aria-label", `Tres vectores tridimensionales desde un origen. Ángulos ${summaries.join(", ")}. Gira la vista arrastrando el gráfico.`);
  }

  function setTriPreset(key) {
    const preset = triPresets[key];
    if (!preset) return;
    for (const name of ["A", "B", "C"]) {
      byId(`triSlider${name}`).value = String(preset[name][0]);
      byId(`triElev${name}`).value = String(preset[name][1]);
    }
    for (const button of document.querySelectorAll("[data-tri-preset]")) button.setAttribute("aria-pressed", String(button.dataset.triPreset === key));
    updateTriVectors();
  }

  function setScanView(key) {
    const view = tomographyViews[key];
    if (!view) return;
    for (const button of document.querySelectorAll("[data-scan-view]")) {
      const active = button.dataset.scanView === key;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
    }
    for (const visual of document.querySelectorAll("[data-scan-visual]")) {
      if (visual.dataset.scanVisual === key) visual.removeAttribute("hidden");
      else visual.setAttribute("hidden", "");
    }
    byId("scanTitle").textContent = view.title;
    byId("scanDescription").textContent = view.description;
    byId("scanQuestion").textContent = view.question;
    byId("scanSvg").setAttribute("aria-label", view.alt);
  }

  function companyType(label) {
    return companyTypeNames[label] || label.replaceAll("_", " ");
  }

  function imagePath(document) {
    const path = String(document.image || "");
    return path.startsWith("images/") && !path.includes("..") ? `data/empresa/${path}` : "";
  }

  async function loadCompanyData() {
    try {
      const [manifestResponse, vectorResponse] = await Promise.all([
        fetch("data/empresa/manifest.json", { cache: "no-store" }),
        fetch("data/empresa/embeddings.json", { cache: "no-store" })
      ]);
      if (!manifestResponse.ok || !vectorResponse.ok) throw new Error("Archivos de datos no disponibles");
      const [manifest, vectors] = await Promise.all([manifestResponse.json(), vectorResponse.json()]);
      if (!Array.isArray(manifest.documents)) throw new Error("Lista de documentos inválida");
      companyDocs = manifest.documents.map((doc) => {
        const vector = vectors[doc.id]?.image;
        if (!Array.isArray(vector) || vector.length !== 2048) return null;
        const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
        return norm > 0 ? { ...doc, vector, norm } : null;
      }).filter(Boolean);
      if (companyDocs.length < 2) throw new Error("Faltan vectores completos");
      const select = byId("treeRootSelect");
      const groups = new Map();
      for (const doc of companyDocs) {
        if (!groups.has(doc.label)) groups.set(doc.label, []);
        groups.get(doc.label).push(doc);
      }
      select.innerHTML = [...groups.entries()].sort((a, b) => companyType(a[0]).localeCompare(companyType(b[0]), "es"))
        .map(([label, docs]) => `<optgroup label="${escapeHtml(companyType(label))}">${docs.map((doc) => `<option value="${escapeHtml(doc.id)}">${escapeHtml(companyType(label))} · ${escapeHtml(doc.id)}</option>`).join("")}</optgroup>`).join("");
      select.disabled = false;
      select.value = companyDocs.some((doc) => doc.id === "photo_000") ? "photo_000" : companyDocs[0].id;
      renderCompanyTree();
    } catch {
      byId("treeStatus").textContent = "Datos no disponibles en esta vista";
      byId("realTree").innerHTML = '<div class="tree-loading">Abre el catálogo con <code>./abrir-lab.sh</code> para cargar las imágenes y vectores locales.</div>';
    }
  }

  function renderCompanyTree() {
    const root = companyDocs.find((doc) => doc.id === byId("treeRootSelect").value);
    if (!root) return;
    const cosine = (a, b) => {
      let product = 0;
      for (let i = 0; i < a.vector.length; i++) product += a.vector[i] * b.vector[i];
      return product / (a.norm * b.norm);
    };
    const neighbors = companyDocs.filter((doc) => doc.id !== root.id)
      .map((doc) => ({ ...doc, score: cosine(root, doc) }))
      .sort((a, b) => b.score - a.score).slice(0, 8);
    const groups = new Map();
    for (const doc of neighbors) {
      if (!groups.has(doc.label)) groups.set(doc.label, []);
      groups.get(doc.label).push(doc);
    }
    byId("treeStatus").textContent = `${companyDocs.length} páginas · ${new Set(companyDocs.map((doc) => doc.label)).size} tipos · 8 vecinos`;
    byId("realTree").innerHTML = `<div class="tree-root-card"><span class="tree-node-kicker">DOCUMENTO RAÍZ</span><div class="tree-root-image"><img src="${escapeHtml(imagePath(root))}" alt="Vista previa del documento ${escapeHtml(root.id)}" loading="lazy"></div><strong>${escapeHtml(companyType(root.label))}</strong><small>${escapeHtml(root.id)}</small><span class="tree-root-hint">Elige otra hoja para seguir el árbol.</span></div><div class="tree-link-rail" aria-hidden="true"><span></span></div><div class="tree-branches">${[...groups.entries()].map(([label, docs]) => `<div class="tree-branch ${label === root.label ? "same-type" : ""}"><div class="tree-branch-title"><span class="tree-branch-dot"></span><strong>${escapeHtml(companyType(label))}</strong><small>${docs.length} ${docs.length === 1 ? "vecino" : "vecinos"}</small></div><div class="tree-leaves">${docs.map((doc) => `<button type="button" class="tree-leaf" data-tree-root="${escapeHtml(doc.id)}" aria-label="Tomar ${escapeHtml(doc.id)} como raíz; coseno ${triFormat(doc.score)}"><img src="${escapeHtml(imagePath(doc))}" alt="" loading="lazy"><span class="tree-leaf-name"><strong>${escapeHtml(doc.id)}</strong><small>${escapeHtml(companyType(doc.label))}</small></span><span class="tree-leaf-score">cos ${triFormat(doc.score)}</span></button>`).join("")}</div></div>`).join("")}</div>`;
  }

  const toyItems = [
    { id: "gato-texto", label: "«un gato»", type: "text", angle: 15, x: 66, y: 19 },
    { id: "gato-imagen", label: "foto gato", type: "image", angle: 23, x: 79, y: 38 },
    { id: "perro-texto", label: "«un perro»", type: "text", angle: 84, x: 40, y: 65 },
    { id: "perro-imagen", label: "foto perro", type: "image", angle: 91, x: 66, y: 78 },
    { id: "casa-texto", label: "«una casa»", type: "text", angle: 158, x: 20, y: 18 },
    { id: "casa-imagen", label: "foto casa", type: "image", angle: 165, x: 24, y: 42 }
  ];

  function updateToy() {
    const query = toyItems.find((item) => item.id === byId("toyQuery").value) || toyItems[0];
    byId("toyMap").innerHTML = toyItems.map((item) =>
      `<div class="toy-point ${item.type === "image" ? "image" : "text"} ${item.id === query.id ? "query" : ""}" style="left:${item.x}%;top:${item.y}%" title="${escapeHtml(item.label)}"><span aria-hidden="true">${item.type === "image" ? "▧" : "T"}</span>${escapeHtml(item.label)}</div>`
    ).join("");
    const ranking = toyItems.filter((item) => item.id !== query.id)
      .map((item) => ({ ...item, score: Math.cos((item.angle - query.angle) * Math.PI / 180) }))
      .sort((a, b) => b.score - a.score);
    byId("toyRanking").innerHTML = ranking.slice(0, 4).map((item, index) =>
      `<div class="ranking-item"><span class="rank-num">${index + 1}.</span><span class="rank-name">${escapeHtml(item.label)}</span><span class="rank-bar" aria-hidden="true"><span style="width:${Math.max(0, item.score) * 100}%"></span></span><span class="rank-score">${item.score.toFixed(3).replace(".", ",")}</span></div>`
    ).join("");
  }

  const noteTypeLabels = { idea: "IDEA", pregunta: "PREGUNTA", hallazgo: "HALLAZGO" };

  function renderNotes() {
    byId("noteCount").textContent = String(notes.length);
    if (!notes.length) {
      byId("notesList").innerHTML = '<div class="notes-empty"><span aria-hidden="true">✎</span>Aún no hay notas.<br>Empieza por una idea que te llamó la atención.</div>';
      return;
    }
    byId("notesList").innerHTML = notes.map((note) => {
      const date = new Date(note.createdAt);
      return `<article class="note-card"><div class="note-meta"><span class="note-type ${escapeHtml(note.type)}">${escapeHtml(noteTypeLabels[note.type] || "NOTA")}</span><time datetime="${escapeHtml(note.createdAt)}">${Number.isNaN(date.getTime()) ? "" : date.toLocaleString("es-CO", { dateStyle: "medium", timeStyle: "short" })}</time></div><p>${escapeHtml(note.text)}</p>${note.next ? `<div class="note-next"><strong>Probar:</strong> ${escapeHtml(note.next)}</div>` : ""}<div class="note-footer"><span>${escapeHtml(topicLabel(note.concept))}</span><button class="text-button" type="button" data-delete-note="${escapeHtml(note.id)}" aria-label="Eliminar esta nota">Eliminar</button></div></article>`;
    }).join("");
  }

  function makeNote(event) {
    event.preventDefault();
    const text = byId("noteText").value.trim();
    if (!text) return;
    notes.unshift({
      id: globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(36).slice(2)}`,
      type: byId("noteType").value,
      concept: byId("noteConcept").value,
      text,
      next: byId("noteNext").value.trim(),
      createdAt: new Date().toISOString()
    });
    saveStored(STORE_NOTES, notes);
    renderNotes();
    byId("noteForm").reset();
    showToast("Nota guardada en este navegador.");
  }

  function download(name, type, content) {
    const url = URL.createObjectURL(new Blob([content], { type }));
    const link = document.createElement("a");
    link.href = url;
    link.download = name;
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function exportNotes(format) {
    if (!notes.length) return showToast("Agrega una nota antes de exportar.");
    const date = new Date().toISOString().slice(0, 10);
    if (format === "json") {
      download(`atlas-multimodal-${date}.json`, "application/json;charset=utf-8", JSON.stringify({ version: 1, exportedAt: new Date().toISOString(), notes }, null, 2));
    } else {
      const text = ["# Mi bitácora multimodal", "", `Exportada: ${date}`, "", ...notes.flatMap((note) => [
        `## ${noteTypeLabels[note.type] || "NOTA"} · ${topicLabel(note.concept)}`,
        "", `Fecha: ${new Date(note.createdAt).toLocaleString("es-CO")}`, "", note.text, "",
        ...(note.next ? [`**Qué quiero probar:** ${note.next}`, ""] : []), "---", ""
      ])].join("\n");
      download(`atlas-multimodal-${date}.md`, "text/markdown;charset=utf-8", text);
    }
    showToast("Bitácora descargada.");
  }

  function beginNote(conceptId) {
    byId("conceptDialog").close();
    byId("noteConcept").value = conceptId;
    byId("bitacora").scrollIntoView({ behavior: "smooth" });
    setTimeout(() => byId("noteText").focus({ preventScroll: true }), 300);
  }

  async function checkLab() {
    const status = byId("connectionStatus");
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 2800);
    try {
      // no-cors comprueba conectividad sin necesitar leer la respuesta de Streamlit.
      await fetch(`${LAB_URL}_stcore/health`, { mode: "no-cors", cache: "no-store", signal: controller.signal });
      status.className = "connection available";
      status.lastElementChild.textContent = "Laboratorio local disponible";
    } catch {
      // Un navegador puede bloquear la prueba incluso si Streamlit sí funciona.
      status.className = "connection unknown";
      status.lastElementChild.textContent = "Estado sin confirmar · prueba abrirlo";
    } finally {
      clearTimeout(timeout);
    }
  }

  function setupNavigation() {
    const links = [...document.querySelectorAll(".nav-link")];
    const sections = [...document.querySelectorAll("#atlasView > section")];
    const observer = new IntersectionObserver(() => {
      if (!byId("labView").hidden) return;
      const threshold = Math.min(280, window.innerHeight * 0.34);
      const active = [...sections].reverse().find((section) => section.getBoundingClientRect().top <= threshold) || sections[0];
      lastAtlasHash = `#${active.id}`;
      for (const link of links) link.classList.toggle("active", link.hash === `#${active.id}`);
    }, { rootMargin: "-20% 0px -65% 0px", threshold: 0 });
    sections.forEach((section) => observer.observe(section));
  }

  function labFrameUrl(module) {
    const url = new URL(LAB_URL);
    if (module.query) url.searchParams.set("modulo", module.query);
    url.searchParams.set("embed", "true");
    url.searchParams.set("shell", "1");
    return url.href;
  }

  function applyRoute(initial = false) {
    const hash = decodeURIComponent(location.hash.slice(1));
    const labKey = hash.startsWith("lab/") ? hash.slice(4) : "";
    const module = LAB_MODULES[labKey];
    const wasLab = !byId("labView").hidden;
    const topLink = byId("topLabLink");
    if (module) {
      if (byId("conceptDialog").open) byId("conceptDialog").close();
      byId("atlasView").hidden = true;
      byId("labView").hidden = false;
      document.body.classList.add("lab-mode");
      byId("labTitle").textContent = module.title;
      byId("labDescription").textContent = module.description;
      byId("labBack").href = lastAtlasHash;
      topLink.href = lastAtlasHash;
      topLink.innerHTML = 'Volver al atlas <span aria-hidden="true">←</span>';
      document.title = `${module.title} · Atlas multimodal`;
      for (const link of document.querySelectorAll(".nav-link")) link.classList.toggle("active", link.classList.contains("nav-external"));
      for (const link of document.querySelectorAll(".lab-module-link")) {
        const active = link.hash === `#lab/${labKey}`;
        link.classList.toggle("active", active);
        if (active) link.setAttribute("aria-current", "page");
        else link.removeAttribute("aria-current");
      }
      const frame = byId("labFrame");
      const url = labFrameUrl(module);
      frame.title = `${module.title} en Streamlit`;
      if (frame.src !== url) {
        byId("labFrameStatus").hidden = false;
        frame.src = url;
      }
      window.scrollTo({ top: 0, behavior: "instant" });
      if (!initial) byId("labTitle").focus({ preventScroll: true });
      return;
    }

    byId("labView").hidden = true;
    byId("atlasView").hidden = false;
    document.body.classList.remove("lab-mode");
    topLink.href = "#lab/inicio";
    topLink.innerHTML = 'Abrir laboratorio <span aria-hidden="true">→</span>';
    document.title = "Atlas multimodal · laboratorio de embeddings";
    const section = ATLAS_SECTIONS.has(hash) ? hash : "inicio";
    if (ATLAS_SECTIONS.has(hash)) lastAtlasHash = `#${hash}`;
    for (const link of document.querySelectorAll(".nav-link")) link.classList.toggle("active", link.hash === `#${section}`);
    if (wasLab || initial && hash && hash !== "contenido") {
      requestAnimationFrame(() => byId(section).scrollIntoView({ behavior: "instant", block: "start" }));
    }
  }

  function setupRoutes() {
    byId("labFrame").addEventListener("load", () => { byId("labFrameStatus").hidden = true; });
    window.addEventListener("hashchange", () => applyRoute());
    applyRoute(true);
  }

  function setupEvents() {
    byId("conceptSearch").addEventListener("input", renderConcepts);
    document.addEventListener("click", (event) => {
      const link = event.target.closest("a[href]");
      if (link) {
        const url = new URL(link.href);
        if (url.origin === new URL(LAB_URL).origin) {
          event.preventDefault();
          const module = url.searchParams.get("modulo") || "inicio";
          location.hash = `#lab/${LAB_MODULES[module] ? module : "inicio"}`;
        }
      }
      const conceptButton = event.target.closest("[data-concept]");
      if (conceptButton) openConcept(conceptButton.dataset.concept);
      const noteButton = event.target.closest("[data-note-concept]");
      if (noteButton) beginNote(noteButton.dataset.noteConcept);
      const deleteButton = event.target.closest("[data-delete-note]");
      if (deleteButton) {
        notes = notes.filter((note) => note.id !== deleteButton.dataset.deleteNote);
        saveStored(STORE_NOTES, notes);
        renderNotes();
        showToast("Nota eliminada.");
      }
    });
    byId("closeDialog").addEventListener("click", () => byId("conceptDialog").close());
    byId("conceptDialog").addEventListener("click", (event) => {
      if (event.target === byId("conceptDialog")) byId("conceptDialog").close();
    });
    byId("practiceGrid").addEventListener("change", (event) => {
      if (!event.target.matches("[data-practice]")) return;
      completed[event.target.dataset.practice] = event.target.checked;
      saveStored(STORE_PROGRESS, completed);
      updateProgress();
    });
    byId("angleSlider").addEventListener("input", updateCosine);
    byId("lengthSlider").addEventListener("input", updateCosine);
    byId("toyQuery").addEventListener("change", updateToy);
    byId("treeRootSelect").addEventListener("change", renderCompanyTree);
    byId("realTree").addEventListener("click", (event) => {
      const leaf = event.target.closest("[data-tree-root]");
      if (!leaf) return;
      byId("treeRootSelect").value = leaf.dataset.treeRoot;
      renderCompanyTree();
    });
    for (const name of ["A", "B", "C"]) {
      for (const id of [`triSlider${name}`, `triElev${name}`]) byId(id).addEventListener("input", () => {
        for (const button of document.querySelectorAll("[data-tri-preset]")) button.setAttribute("aria-pressed", "false");
        updateTriVectors();
      });
    }
    document.querySelector(".tri-presets").addEventListener("click", (event) => {
      const button = event.target.closest("[data-tri-preset]");
      if (button) setTriPreset(button.dataset.triPreset);
    });
    document.querySelector(".tri-camera").addEventListener("click", (event) => {
      const button = event.target.closest("[data-camera]");
      if (!button) return;
      const action = button.dataset.camera;
      if (action === "reset") { triCamera.yaw = 35; triCamera.pitch = 28; }
      if (action === "left") triCamera.yaw -= 18;
      if (action === "right") triCamera.yaw += 18;
      if (action === "up") triCamera.pitch = Math.min(70, triCamera.pitch + 12);
      if (action === "down") triCamera.pitch = Math.max(-70, triCamera.pitch - 12);
      updateTriVectors();
    });
    let triDrag = null;
    byId("triSvg").addEventListener("pointerdown", (event) => {
      triDrag = { pointerId: event.pointerId, x: event.clientX, y: event.clientY, yaw: triCamera.yaw, pitch: triCamera.pitch };
      byId("triSvg").setPointerCapture(event.pointerId);
      byId("triSvg").classList.add("dragging");
    });
    byId("triSvg").addEventListener("pointermove", (event) => {
      if (!triDrag || triDrag.pointerId !== event.pointerId) return;
      triCamera.yaw = triDrag.yaw + (event.clientX - triDrag.x) * 0.65;
      triCamera.pitch = Math.max(-70, Math.min(70, triDrag.pitch - (event.clientY - triDrag.y) * 0.5));
      updateTriVectors();
    });
    for (const type of ["pointerup", "pointercancel", "lostpointercapture"]) byId("triSvg").addEventListener(type, () => {
      triDrag = null;
      byId("triSvg").classList.remove("dragging");
    });
    document.querySelector(".scan-controls").addEventListener("click", (event) => {
      const button = event.target.closest("[data-scan-view]");
      if (button) setScanView(button.dataset.scanView);
    });
    byId("noteForm").addEventListener("submit", makeNote);
    byId("exportMd").addEventListener("click", () => exportNotes("md"));
    byId("exportJson").addEventListener("click", () => exportNotes("json"));
    byId("copyTunnel").addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText("./abrir-lab.sh");
        showToast("Comando copiado.");
      } catch {
        showToast("Copia el comando que aparece a la izquierda.");
      }
    });
  }

  function init() {
    renderFilters();
    renderConcepts();
    renderPractices();
    renderGlossary();
    renderNotes();
    byId("noteConcept").insertAdjacentHTML("beforeend", concepts.map((concept) => `<option value="${concept.id}">${escapeHtml(concept.title)}</option>`).join("") + '<option value="tomografia">Tomografía · entrenamiento y predicción</option>');
    updateCosine();
    setTriPreset("separados");
    updateToy();
    setScanView("capas");
    setupEvents();
    setupNavigation();
    setupRoutes();
    loadCompanyData();
    checkLab();
    setInterval(checkLab, 20000);
  }

  init();
})();
