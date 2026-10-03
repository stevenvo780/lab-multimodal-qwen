"""Tomografía didáctica: capas reales de Qwen y entrenamiento de una sonda local.

Qwen3-VL-Embedding permanece congelado en kratos. Solo entrenamos un clasificador
lineal pequeño con los embeddings finales, en la CPU del equipo local.
"""

from __future__ import annotations

import base64
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss

from gpu_client import GPUClient, GPUClientError, encode_image


ROOT = Path(__file__).resolve().parent
CLASSES = ("Animales", "Comida", "Lugares")
COLORS = {"Animales": "#23b899", "Comida": "#eea443", "Lugares": "#9b84ee"}
PAIRS = (
    ("perro", "Animales", "perro.jpg", "un perro", False),
    ("gato", "Animales", "gato.jpg", "un gato", False),
    ("león", "Animales", "leon.jpg", "un león", False),
    ("panda", "Animales", "panda.jpg", "un panda", True),
    ("pizza", "Comida", "pizza.jpg", "una pizza", False),
    ("banano", "Comida", "banano.jpg", "un banano", False),
    ("café", "Comida", "cafe.jpg", "una taza de café", False),
    ("manzana", "Comida", "manzana.jpg", "una manzana", True),
    ("Coliseo", "Lugares", "coliseo.jpg", "el Coliseo de Roma", False),
    ("Torre Eiffel", "Lugares", "torre-eiffel.jpg", "la Torre Eiffel de París", False),
    ("Everest", "Lugares", "everest.jpg", "el monte Everest", True),
)


@dataclass(frozen=True)
class Example:
    name: str
    category: str
    modality: str
    held_out: bool
    text: str = ""
    image: bytes = b""

    def api_item(self) -> dict[str, str]:
        if self.image:
            return {"image_base64": encode_image(self.image)}
        return {"text": self.text}


def examples_from_samples() -> tuple[Example, ...]:
    examples = []
    for name, category, filename, phrase, held_out in PAIRS:
        path = ROOT / "samples" / filename
        if not path.exists():
            continue
        examples.append(Example(name, category, "texto", held_out, text=phrase))
        examples.append(Example(name, category, "imagen", held_out, image=path.read_bytes()))
    return tuple(examples)


def dataset_signature(examples: tuple[Example, ...], base_url: str) -> str:
    digest = hashlib.sha256(base_url.encode())
    for item in examples:
        for value in (item.name, item.category, item.modality, item.text, str(item.held_out)):
            digest.update(value.encode())
        digest.update(item.image)
    return digest.hexdigest()


def normalized(vectors: list[list[float]]) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 2048:
        raise GPUClientError("La API no devolvió vectores de 2048 dimensiones.")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise GPUClientError("Un vector de la API tiene norma casi nula.")
    return values / norms


def train_probe(
    vectors: np.ndarray, examples: tuple[Example, ...], epochs: int
) -> tuple[SGDClassifier, dict[str, list[float]], np.ndarray, np.ndarray]:
    labels = np.asarray([CLASSES.index(item.category) for item in examples], dtype=int)
    train_mask = np.asarray([not item.held_out for item in examples], dtype=bool)
    x_train, y_train = vectors[train_mask], labels[train_mask]
    x_test, y_test = vectors[~train_mask], labels[~train_mask]
    model = SGDClassifier(
        loss="log_loss", penalty="l2", alpha=0.001,
        learning_rate="constant", eta0=0.5, random_state=42,
        fit_intercept=True, shuffle=False,
    )
    history = {"train_loss": [], "test_loss": [], "train_accuracy": [], "test_accuracy": []}
    for epoch in range(epochs):
        order = np.random.default_rng(1000 + epoch).permutation(len(y_train))
        model.partial_fit(x_train[order], y_train[order], classes=np.arange(len(CLASSES)))
        for split, x, y in (("train", x_train, y_train), ("test", x_test, y_test)):
            probs = model.predict_proba(x)
            history[f"{split}_loss"].append(float(log_loss(y, probs, labels=np.arange(len(CLASSES)))))
            history[f"{split}_accuracy"].append(float(np.mean(model.predict(x) == y)))
    return model, history, train_mask, labels


def loss_figure(history: dict[str, list[float]]) -> go.Figure:
    epoch_axis = list(range(1, len(history["train_loss"]) + 1))
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=epoch_axis, y=history["train_loss"], mode="lines",
                                name="Ejemplos de entrenamiento", line={"color": "#23b899", "width": 3}))
    figure.add_trace(go.Scatter(x=epoch_axis, y=history["test_loss"], mode="lines",
                                name="Conceptos reservados", line={"color": "#eea443", "width": 3}))
    figure.update_layout(title="Pérdida durante el entrenamiento de la sonda",
                         xaxis_title="Época", yaxis_title="Pérdida logarítmica (menor es mejor)",
                         height=340, margin={"l": 30, "r": 20, "t": 65, "b": 50})
    return figure


def map_figure(
    vectors: np.ndarray, examples: tuple[Example, ...],
    train_mask: np.ndarray, query: np.ndarray | None = None,
) -> tuple[go.Figure, float]:
    pca = PCA(n_components=2)
    pca.fit(vectors[train_mask])
    points = pca.transform(vectors)
    figure = go.Figure()
    for category in CLASSES:
        for modality in ("texto", "imagen"):
            for held_out in (False, True):
                indexes = [index for index, item in enumerate(examples)
                           if item.category == category and item.modality == modality
                           and item.held_out == held_out]
                if not indexes:
                    continue
                figure.add_trace(go.Scatter(
                    x=points[indexes, 0], y=points[indexes, 1],
                    mode="markers", name=f"{category} · {modality}{' · prueba' if held_out else ''}",
                    marker={"size": 13, "color": COLORS[category],
                            "symbol": ("circle" if modality == "texto" else "square")
                            + ("-open" if held_out else "")},
                    customdata=[examples[index].name for index in indexes],
                    hovertemplate="%{customdata}<br>x=%{x:.2f}, y=%{y:.2f}<extra></extra>",
                ))
    if query is not None:
        point = pca.transform(query.reshape(1, -1))[0]
        figure.add_trace(go.Scatter(x=[point[0]], y=[point[1]], mode="markers+text",
                                    name="Nueva entrada", text=["★ Predicción"],
                                    textposition="top center",
                                    marker={"size": 19, "color": "#e9478d", "symbol": "star"}))
    figure.update_layout(title="Mapa de embeddings finales (PCA)",
                         xaxis_title="Componente 1", yaxis_title="Componente 2",
                         height=520, margin={"l": 30, "r": 20, "t": 65, "b": 50},
                         legend={"font": {"size": 10}})
    return figure, float(pca.explained_variance_ratio_.sum())


def prediction_figure(scores: np.ndarray) -> go.Figure:
    figure = go.Figure(go.Bar(
        x=scores, y=list(CLASSES), orientation="h",
        marker_color=[COLORS[name] for name in CLASSES],
        text=[f"{score:.2f}" for score in scores], textposition="outside",
    ))
    figure.update_layout(title="Salida de la sonda para la nueva entrada",
                         xaxis={"range": [0, 1.08], "title": "Puntuación relativa"},
                         yaxis={"autorange": "reversed"},
                         height=290, margin={"l": 20, "r": 30, "t": 60, "b": 45})
    return figure


def prepare_scan_image(data: bytes) -> bytes:
    try:
        with Image.open(io.BytesIO(data)) as image:
            image = image.convert("RGB")
            image.thumbnail((512, 512))
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=82, optimize=True)
            result = output.getvalue()
    except (OSError, ValueError) as exc:
        raise GPUClientError("La imagen de inspección no se pudo abrir.") from exc
    if len(result) > 1024 * 1024:
        raise GPUClientError("La imagen de inspección supera 1 MiB aun reducida.")
    return result


def scan_chart(scans: dict[str, dict[str, Any]], field: str, title: str, y_title: str) -> go.Figure:
    figure = go.Figure()
    colors = {"Texto": "#23b899", "Imagen": "#9b84ee"}
    for label, result in scans.items():
        layers = result["layers"]
        figure.add_trace(go.Scatter(
            x=[entry["layer"] for entry in layers],
            y=[entry[field] for entry in layers],
            mode="lines+markers", name=label,
            line={"color": colors[label], "width": 2.5}, marker={"size": 4},
        ))
    figure.update_layout(title=title, xaxis_title="Capa del decodificador", yaxis_title=y_title,
                         height=320, margin={"l": 35, "r": 20, "t": 65, "b": 45})
    return figure


def render_layer_scan(client: GPUClient) -> None:
    st.divider()
    st.header("Tomografía interna · 28 capas")
    st.write("Seguimos el **token de resumen** por las capas del decodificador. La imagen entra antes por el codificador visual e influye en esta trayectoria. Estas curvas describen activaciones; no revelan por sí solas qué 'piensa' el modelo.")
    col_text, col_image = st.columns(2)
    with col_text:
        phrase = st.text_input("Frase para inspeccionar", "un perro", max_chars=1000,
                               key="scan_text")
    with col_image:
        sample_names = ["perro.jpg", "gato.jpg", "pizza.jpg", "everest.jpg"]
        selected = st.selectbox("Imagen para inspeccionar", sample_names, key="scan_sample")
        uploaded = st.file_uploader("O sube una imagen", type=["jpg", "jpeg", "png", "webp"],
                                    key="scan_upload")
    run_scan = st.button("Inspeccionar texto e imagen", type="primary", key="scan_run")
    # Restore only after preparing the same request successfully. Even an
    # unexpected error or interruption must leave the previous scan inactive.
    saved = st.session_state.pop("tomografia_scans", None)
    try:
        image_bytes = uploaded.getvalue() if uploaded else (ROOT / "samples" / selected).read_bytes()
        compact_image = prepare_scan_image(image_bytes)
        phrase = phrase.strip()
        request = (phrase, hashlib.sha256(compact_image).hexdigest(), client.base_url)
    except (GPUClientError, OSError, ValueError, Image.DecompressionBombError) as exc:
        st.error(str(exc))
        return
    if not run_scan and saved and saved.get("request") == request:
        # Retries, changed inputs and legacy results remain invalidated.
        st.session_state["tomografia_scans"] = saved
    if run_scan:
        try:
            scans = {}
            with st.spinner("Kratos recorre las capas; puede tardar unos segundos…"):
                if phrase:
                    scans["Texto"] = client.inspect({"text": phrase})
                scans["Imagen"] = client.inspect({"image_base64": encode_image(compact_image)})
            models = {label: result.get("model") if isinstance(result.get("model"), str) else None
                      for label, result in scans.items()}
            if len({model for model in models.values() if model}) > 1:
                raise GPUClientError("El modelo cambió durante la inspección. Vuelve a intentarlo.")
            # Publish both modalities together, keeping only identity the API supplied.
            st.session_state["tomografia_scans"] = {
                "request": request, "scans": scans, "models": models,
            }
        except (GPUClientError, OSError) as exc:
            st.error(str(exc))
    saved = st.session_state.get("tomografia_scans")
    scans = saved["scans"] if saved and saved.get("request") == request else None
    if scans:
        metrics = st.columns(len(scans))
        for column, (label, result) in zip(metrics, scans.items()):
            column.metric(f"Tokens · {label.lower()}", result.get("input_token_count", "?"))
        left, right = st.columns(2)
        with left:
            st.plotly_chart(scan_chart(scans, "rms", "Magnitud RMS del token por capa", "RMS"),
                            width="stretch")
        with right:
            st.plotly_chart(scan_chart(scans, "delta_cos", "Dirección entre capas consecutivas",
                                          "Coseno con la capa anterior"), width="stretch")
        st.caption("Un coseno menor indica un cambio mayor de dirección. La escala del RMS puede diferir entre texto e imagen; la similitud semántica se evalúa con el embedding final, no con cada capa aislada.")


def render_tomography(client: GPUClient) -> None:
    st.title("Tomografía · entrenamiento y predicción")
    st.caption("Embeddings reales en la GPU de kratos · sonda lineal entrenada en la CPU local · Qwen congelado")
    st.markdown("[← Volver a los cuatro experimentos](http://127.0.0.1:8501/)")
    st.info("Aquí entrenamos **una sonda pequeña**, no Qwen3-VL. Sus entradas son vectores de 2048 dimensiones que Qwen ya sabe producir. El preentrenamiento multimodal de Qwen usa objetivos contrastivos; este ejercicio muestra en pequeño qué es ajustar pesos y luego predecir.")

    examples = examples_from_samples()
    if len(examples) != len(PAIRS) * 2:
        st.error("Faltan imágenes de ejemplo en samples/. No se puede construir el conjunto didáctico.")
        render_layer_scan(client)
        return
    signature = dataset_signature(examples, client.base_url)
    st.header("1 · Preparar datos multimodales")
    st.write("Usamos 11 parejas frase–imagen de tres categorías. Reservamos **panda, manzana y Everest**, tanto en texto como en imagen, para probar conceptos que la sonda no vio durante el entrenamiento.")
    with st.expander("Ver los 22 ejemplos y su partición"):
        st.dataframe([{"Concepto": item.name, "Clase": item.category,
                       "Modalidad": item.modality,
                       "Uso": "Prueba" if item.held_out else "Entrenamiento"}
                      for item in examples], hide_index=True, width="stretch")
    if st.button("Obtener 22 embeddings reales", type="primary", key="probe_embed"):
        try:
            with st.spinner("La GPU de kratos codifica texto e imágenes…"):
                vectors = normalized(client.embed([item.api_item() for item in examples]))
            st.session_state["probe_vectors"] = {"signature": signature, "vectors": vectors}
        except GPUClientError as exc:
            st.error(str(exc))
    saved = st.session_state.get("probe_vectors")
    vectors = saved["vectors"] if saved and saved["signature"] == signature else None

    st.header("2 · Entrenar y mirar el mapa")
    epochs = st.slider("Épocas de la sonda", 5, 120, 45, 5, key="probe_epochs")
    st.code("embedding = Qwen(texto_o_imagen)  # Qwen congelado\nclase = softmax(W @ embedding + b)  # W y b sí se entrenan", language="python")
    if vectors is None:
        st.info("Primero pulsa «Obtener 22 embeddings reales». Entonces aparecerán la curva de entrenamiento y el mapa PCA.")
    else:
        model, history, train_mask, labels = train_probe(vectors, examples, epochs)
        test_mask = ~train_mask
        test_predictions = model.predict(vectors[test_mask])
        by_modality = {}
        for modality in ("texto", "imagen"):
            selected = np.asarray([item.modality == modality for item in examples]) & test_mask
            by_modality[modality] = int(np.sum(model.predict(vectors[selected]) == labels[selected]))
        col1, col2, col3 = st.columns(3)
        col1.metric("Ejemplos para aprender", int(train_mask.sum()))
        col2.metric("Prueba · texto", f"{by_modality['texto']}/3")
        col3.metric("Prueba · imagen", f"{by_modality['imagen']}/3")
        st.plotly_chart(loss_figure(history), width="stretch")
        st.caption("La curva naranja usa conceptos reservados. Son solo seis casos; no es una evaluación de calidad general del modelo.")

        st.header("3 · Predecir con texto, imagen o ambos")
        phrase = st.text_input("Frase nueva (opcional)", "un animal en la naturaleza", key="probe_query")
        upload = st.file_uploader("Imagen nueva (opcional)", type=["jpg", "jpeg", "png", "webp"],
                                  key="probe_image")
        image_data = upload.getvalue() if upload else b""
        if st.button("Predecir con la sonda", type="primary", key="probe_predict"):
            if not phrase.strip() and not image_data:
                st.error("Escribe una frase o sube una imagen.")
            elif len(image_data) > 8 * 1024 * 1024:
                st.error("La imagen debe ocupar 8 MiB o menos.")
            else:
                try:
                    item = {}
                    if phrase.strip():
                        item["text"] = phrase.strip()
                    if image_data:
                        item["image_base64"] = encode_image(image_data)
                    with st.spinner("Qwen genera el embedding; la sonda calcula la clase…"):
                        query = normalized(client.embed([item]))[0]
                    key = hashlib.sha256((signature + phrase.strip()).encode() + image_data).hexdigest()
                    st.session_state["probe_prediction"] = {"key": key, "vector": query}
                except GPUClientError as exc:
                    st.error(str(exc))
        prediction = st.session_state.get("probe_prediction")
        current_key = hashlib.sha256((signature + phrase.strip()).encode() + image_data).hexdigest()
        query = prediction["vector"] if prediction and prediction["key"] == current_key else None
        if query is not None:
            scores = model.predict_proba(query.reshape(1, -1))[0]
            st.plotly_chart(prediction_figure(scores), width="stretch")
            st.caption("Puntuaciones de una sonda entrenada con muy pocos ejemplos; no son probabilidades calibradas ni una descripción exhaustiva de la entrada.")
            similarity = vectors[train_mask] @ query
            nearest = np.argsort(similarity)[::-1][:5]
            train_examples = [item for item in examples if not item.held_out]
            st.dataframe([{"Vecino": train_examples[int(index)].name,
                           "Clase": train_examples[int(index)].category,
                           "Modalidad": train_examples[int(index)].modality,
                           "Coseno": round(float(similarity[int(index)]), 3)}
                          for index in nearest], hide_index=True, width="stretch")
        map_chart, explained = map_figure(vectors, examples, train_mask, query)
        st.plotly_chart(map_chart, width="stretch")
        st.caption(f"PCA conserva {explained:.1%} de la varianza del conjunto de entrenamiento en estos dos ejes. Las distancias del mapa pueden distorsionarse; la sonda aprende en 2048D.")
    render_layer_scan(client)
