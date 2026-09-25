"""Comparador multimodal A/B/C y colección libre para el laboratorio local.

La GPU remota produce embeddings. PCA, geometría y búsqueda corren en este equipo.
Importar este módulo no inicia inferencia ni modifica el laboratorio principal.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from sklearn.decomposition import PCA

from gpu_client import EXPECTED_DIMENSION, GPUClient, GPUClientError, encode_image


MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
MAX_EXTRA_IMAGES = 20
MAX_EXTRA_TEXTS = 30
MAX_TEXT_CHARS = 2_000
PALETTE = {
    "A": "#23b899", "B": "#eea443", "C": "#9b84ee",
    "texto": "#438ed9", "imagen": "#ef6a74", "mixto": "#ad78c7",
}


@dataclass(frozen=True)
class CompareItem:
    label: str
    text: str = ""
    image: bytes = b""

    @property
    def modality(self) -> str:
        if self.text and self.image:
            return "mixto"
        return "imagen" if self.image else "texto"

    def payload(self) -> dict[str, str]:
        item: dict[str, str] = {}
        if self.text:
            item["text"] = self.text
        if self.image:
            item["image_base64"] = encode_image(self.image)
        return item

    def content_key(self, client: GPUClient, instruction: str) -> str:
        digest = hashlib.sha256()
        for part in (client.base_url, instruction, self.text, hashlib.sha256(self.image).hexdigest()):
            digest.update(part.encode("utf-8"))
            digest.update(b"\x00")
        return digest.hexdigest()


def _validate_image(label: str, data: bytes) -> bytes:
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise GPUClientError(f"{label}: la imagen debe medir entre 1 byte y 8 MiB.")
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in {"JPEG", "PNG", "WEBP"}:
                raise GPUClientError(f"{label}: usa JPEG, PNG o WebP.")
            width, height = image.size
            if width * height > MAX_IMAGE_PIXELS:
                raise GPUClientError(f"{label}: la imagen supera 20 megapíxeles.")
            image.verify()
    except (OSError, ValueError) as exc:
        raise GPUClientError(f"{label}: no se pudo leer la imagen.") from exc
    return data


def _from_upload(label: str, uploaded: Any) -> bytes:
    return _validate_image(label, uploaded.getvalue()) if uploaded is not None else b""


def _normalized_matrix(vectors: list[list[float]]) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != EXPECTED_DIMENSION:
        raise GPUClientError("La GPU devolvió embeddings con una forma inesperada.")
    if not np.isfinite(matrix).all():
        raise GPUClientError("La GPU devolvió valores no finitos.")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms < 1e-8):
        raise GPUClientError("La GPU devolvió un vector de longitud casi nula.")
    return matrix / norms[:, None]


def _embed_cached(client: GPUClient, items: list[CompareItem], instruction: str) -> np.ndarray:
    """Guarda vectores por contenido; envía textos en lotes e imágenes de una en una."""
    cache: dict[str, list[float]] = st.session_state.setdefault("comparador_embedding_cache", {})
    unique: dict[str, CompareItem] = {}
    keys = []
    for item in items:
        key = item.content_key(client, instruction)
        keys.append(key)
        if key not in cache:
            unique[key] = item

    text_pending = [(key, item) for key, item in unique.items() if not item.image]
    image_pending = [(key, item) for key, item in unique.items() if item.image]
    for start in range(0, len(text_pending), 4):
        batch = text_pending[start:start + 4]
        vectors = client.embed([item.payload() for _, item in batch], instruction=instruction or None)
        clean = _normalized_matrix(vectors)
        for (key, _), vector in zip(batch, clean, strict=True):
            cache[key] = vector.tolist()
    for key, item in image_pending:
        vector = _normalized_matrix(client.embed([item.payload()], instruction=instruction or None))[0]
        cache[key] = vector.tolist()

    return _normalized_matrix([cache[key] for key in keys])


def _collection_signature(items: list[CompareItem], client: GPUClient, instruction: str) -> str:
    entries = [(item.label, item.content_key(client, instruction)) for item in items]
    return hashlib.sha256(json.dumps(entries, ensure_ascii=False).encode("utf-8")).hexdigest()


def _pca_3d(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    if np.allclose(matrix, matrix[0]):
        return np.zeros((len(matrix), 3)), 0.0
    pca = PCA(n_components=3)
    points = pca.fit_transform(matrix)
    explained = float(np.nansum(pca.explained_variance_ratio_))
    return points, explained


def _pca_figure(points: np.ndarray, items: list[CompareItem]) -> go.Figure:
    figure = go.Figure()
    shown_modalities: set[str] = set()
    for index, item in enumerate(items):
        letter = item.label if index < 3 else ""
        color = PALETTE.get(letter, PALETTE[item.modality])
        show_legend = index < 3 or item.modality not in shown_modalities
        if index >= 3:
            shown_modalities.add(item.modality)
        figure.add_trace(go.Scatter3d(
            x=[points[index, 0]], y=[points[index, 1]], z=[points[index, 2]],
            mode="markers+text", name=item.label if index < 3 else item.modality,
            text=[letter], textposition="top center",
            marker={"size": 9 if index < 3 else 6, "color": color,
                    "symbol": "diamond" if index < 3 else "circle",
                    "line": {"color": "#f8fafc", "width": 1}},
            customdata=[[item.label, item.modality]],
            hovertemplate="%{customdata[0]} · %{customdata[1]}<br>PC1=%{x:.3f}<br>PC2=%{y:.3f}<br>PC3=%{z:.3f}<extra></extra>",
            showlegend=show_legend,
        ))
    figure.update_layout(
        title="Mapa PCA de la colección", height=590,
        scene={"xaxis_title": "PC1", "yaxis_title": "PC2", "zaxis_title": "PC3",
               "aspectmode": "data"},
        margin={"l": 0, "r": 0, "t": 55, "b": 0},
    )
    return figure


def _exact_coordinates(triad: np.ndarray) -> np.ndarray:
    """Tres vectores de 2048D tienen una realización exacta en 3D."""
    gram = triad @ triad.T
    gram = (gram + gram.T) / 2
    values, basis = np.linalg.eigh(gram)
    return basis @ np.diag(np.sqrt(np.maximum(values, 0)))


def _triangle_figure(triad: np.ndarray) -> go.Figure:
    points = _exact_coordinates(triad)
    figure = go.Figure()
    for index, letter in enumerate("ABC"):
        point = points[index]
        figure.add_trace(go.Scatter3d(
            x=[0, point[0]], y=[0, point[1]], z=[0, point[2]],
            mode="lines+markers+text", name=letter,
            line={"color": PALETTE[letter], "width": 6},
            marker={"size": [2, 8], "color": PALETTE[letter]},
            text=["", letter], textposition="top center",
            hovertemplate=f"{letter}<br>x=%{{x:.3f}}<br>y=%{{y:.3f}}<br>z=%{{z:.3f}}<extra></extra>",
        ))
    closed = points[[0, 1, 2, 0]]
    figure.add_trace(go.Scatter3d(
        x=closed[:, 0], y=closed[:, 1], z=closed[:, 2],
        mode="lines", name="Triángulo A–B–C",
        line={"color": "#e2e8f0", "width": 5}, hoverinfo="skip",
    ))
    figure.update_layout(
        title="Triángulo exacto de A, B y C", height=550,
        scene={"xaxis_title": "Eje 1", "yaxis_title": "Eje 2", "zaxis_title": "Eje 3",
               "aspectmode": "cube", "xaxis": {"range": [-1.1, 1.1]},
               "yaxis": {"range": [-1.1, 1.1]}, "zaxis": {"range": [-1.1, 1.1]}},
        margin={"l": 0, "r": 0, "t": 55, "b": 0},
    )
    return figure


def _pair_metrics(triad: np.ndarray) -> list[dict[str, str | float]]:
    rows = []
    for first, second in ((0, 1), (1, 2), (0, 2)):
        dot = float(triad[first] @ triad[second])
        cosine = float(np.clip(dot, -1.0, 1.0))
        rows.append({
            "Par": f"{'ABC'[first]}–{'ABC'[second]}",
            "Producto escalar": round(dot, 4),
            "Coseno": round(cosine, 4),
            "Ángulo desde origen (°)": round(float(np.degrees(np.arccos(cosine))), 2),
            "Distancia 2048D": round(float(np.linalg.norm(triad[first] - triad[second])), 4),
        })
    return rows


def _interior_angles(triad: np.ndarray) -> list[dict[str, str | float]]:
    rows = []
    for index, letter in enumerate("ABC"):
        other = [position for position in range(3) if position != index]
        u, v = triad[other[0]] - triad[index], triad[other[1]] - triad[index]
        denominator = float(np.linalg.norm(u) * np.linalg.norm(v))
        angle = "indefinido" if denominator < 1e-10 else round(
            float(np.degrees(np.arccos(np.clip((u @ v) / denominator, -1.0, 1.0)))), 2
        )
        rows.append({"Vértice": letter, "Ángulo interior (°)": angle})
    return rows


def _thumbnail(data: bytes) -> bytes:
    with Image.open(io.BytesIO(data)) as image:
        image = image.convert("RGB")
        image.thumbnail((220, 160))
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=82)
        return output.getvalue()


def _render_neighbors(items: list[CompareItem], matrix: np.ndarray,
                      query: np.ndarray, excluded: int | None, top_k: int) -> None:
    scores = matrix @ query
    if excluded is not None:
        scores[excluded] = -np.inf
    order = np.argsort(scores)[::-1][:top_k]
    for rank, index in enumerate(order, start=1):
        item = items[int(index)]
        with st.container(border=True):
            preview, details = st.columns([1, 4], vertical_alignment="center")
            with preview:
                if item.image:
                    st.image(_thumbnail(item.image), width=135)
                else:
                    st.write("📝")
            with details:
                st.write(f"{rank}. {item.label}")
                st.caption(f"Coseno real en 2048D: {float(scores[index]):.4f} · {item.modality}")
                if item.text:
                    st.write(item.text[:240])


def _inputs() -> list[CompareItem]:
    st.subheader("A, B y C")
    st.write("Cada ranura puede contener texto, una imagen o ambas cosas. El modelo produce un solo vector por ranura.")
    columns = st.columns(3)
    slots = []
    defaults = {"A": "un perro en la playa", "B": "un gato", "C": "una pizza"}
    for column, letter in zip(columns, "ABC", strict=True):
        with column:
            with st.container(border=True):
                st.markdown(f"### {letter}")
                phrase = st.text_input("Texto", defaults[letter], max_chars=MAX_TEXT_CHARS,
                                       key=f"comparador_{letter}_text")
                upload = st.file_uploader("Imagen", type=["jpg", "jpeg", "png", "webp"],
                                          key=f"comparador_{letter}_image")
                image = _from_upload(letter, upload)
                if not phrase.strip() and not image:
                    raise GPUClientError(f"La ranura {letter} necesita texto, imagen o ambos.")
                slots.append(CompareItem(letter, phrase.strip(), image))

    st.subheader("Ampliar la colección")
    uploads = st.file_uploader(
        "Imágenes adicionales", type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True, key="comparador_extra_images",
    ) or []
    extra_text = st.text_area("Textos adicionales · uno por línea", "", height=110,
                              key="comparador_extra_texts")
    if len(uploads) > MAX_EXTRA_IMAGES:
        raise GPUClientError(f"Añade como máximo {MAX_EXTRA_IMAGES} imágenes adicionales.")
    lines = [line.strip() for line in extra_text.splitlines() if line.strip()]
    if len(lines) > MAX_EXTRA_TEXTS:
        raise GPUClientError(f"Añade como máximo {MAX_EXTRA_TEXTS} textos adicionales.")
    if any(len(line) > MAX_TEXT_CHARS for line in lines):
        raise GPUClientError(f"Cada texto adicional debe tener como máximo {MAX_TEXT_CHARS} caracteres.")
    extras = [CompareItem(f"Imagen {index}: {upload.name}", image=_from_upload(upload.name, upload))
              for index, upload in enumerate(uploads, start=1)]
    extras.extend(CompareItem(f"Texto {index}", text=line)
                  for index, line in enumerate(lines, start=1))
    st.caption(f"Colección: 3 ranuras + {len(uploads)} imágenes + {len(lines)} textos.")
    return slots + extras


def render_comparator(client: GPUClient) -> None:
    """Renderiza el comparador independiente para integrarlo desde laboratorio.py."""
    st.title("Comparador multimodal · colección en 3D")
    st.caption("Qwen3-VL-Embedding-2B en kratos · vectores de 2048 dimensiones · cálculos y gráficos locales")
    st.info("El triángulo A/B/C conserva exactamente sus relaciones en 2048D. El mapa PCA resume toda la colección en tres coordenadas y puede deformar distancias y vecindades.")
    try:
        items = _inputs()
    except GPUClientError as exc:
        st.error(str(exc))
        return

    with st.expander("Instrucción de representación y caché"):
        instruction = st.text_input("Instrucción común a todos los elementos", "",
                                    max_chars=500, key="comparador_instruction",
                                    help="Vacía usa la instrucción predeterminada del modelo.")
        if st.button("Vaciar caché de embeddings", key="comparador_clear_cache"):
            st.session_state.pop("comparador_embedding_cache", None)
            st.session_state.pop("comparador_result", None)
            st.session_state.pop("comparador_query_result", None)
            st.info("La caché de esta sesión quedó vacía.")

    signature = _collection_signature(items, client, instruction)
    if st.button("Calcular colección y mapa", type="primary", key="comparador_calculate"):
        try:
            with st.spinner("Kratos calcula los embeddings nuevos; PCA corre en este equipo…"):
                matrix = _embed_cached(client, items, instruction)
                points, explained = _pca_3d(matrix)
                st.session_state["comparador_result"] = {
                    "signature": signature, "matrix": matrix,
                    "points": points, "explained": explained,
                }
        except (GPUClientError, OSError, ValueError) as exc:
            st.error(str(exc))

    result = st.session_state.get("comparador_result")
    if not result:
        return
    if result["signature"] != signature:
        st.info("Cambiaste la colección o la instrucción. Pulsa «Calcular colección y mapa» para actualizar los gráficos.")
        return

    matrix = result["matrix"]
    st.header("1 · Toda la colección en PCA 3D")
    st.plotly_chart(_pca_figure(result["points"], items), width="stretch")
    st.caption(f"Las tres componentes muestran {result['explained']:.1%} de la varianza de esta colección. Con solo A/B/C, la tercera componente es cero por construcción. Seleccionar un punto o ver cercanía visual no reemplaza el coseno original.")

    st.header("2 · Triángulo exacto de A, B y C")
    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(_triangle_figure(matrix[:3]), width="stretch")
    with right:
        st.markdown("**Relaciones entre vectores desde el origen**")
        st.dataframe(_pair_metrics(matrix[:3]), hide_index=True, width="stretch")
        st.markdown("**Ángulos interiores del triángulo**")
        st.dataframe(_interior_angles(matrix[:3]), hide_index=True, width="stretch")
        st.caption("Los vectores están normalizados: aquí producto escalar y coseno coinciden. La distancia es euclidiana entre extremos en 2048D. Los ejes del dibujo se eligen solo para representarlo.")

    st.header("3 · Consulta y vecinos")
    source = st.radio("Consultar con", ["Elemento de la colección", "Nueva entrada"],
                      horizontal=True, key="comparador_query_source")
    query: np.ndarray | None = None
    excluded: int | None = None
    if source == "Elemento de la colección":
        selected = st.selectbox("Elemento", range(len(items)),
                                format_func=lambda index: f"{items[index].label} · {items[index].modality}",
                                key="comparador_selected")
        exclude_self = st.checkbox("Excluir el propio elemento", value=True,
                                   key="comparador_exclude_self")
        query = matrix[selected]
        excluded = selected if exclude_self else None
    else:
        query_text = st.text_input("Texto de la nueva consulta (opcional)", "",
                                   max_chars=MAX_TEXT_CHARS, key="comparador_query_text")
        query_upload = st.file_uploader("Imagen de la nueva consulta (opcional)",
                                        type=["jpg", "jpeg", "png", "webp"],
                                        key="comparador_query_image")
        try:
            query_image = _from_upload("Consulta", query_upload)
        except GPUClientError as exc:
            st.error(str(exc))
            return
        query_item = CompareItem("Consulta", query_text.strip(), query_image)
        query_signature = query_item.content_key(client, instruction)
        if st.button("Calcular nueva consulta", key="comparador_query_calculate"):
            if not query_item.payload():
                st.error("Escribe texto, sube una imagen o usa ambos en la nueva consulta.")
            else:
                try:
                    with st.spinner("Kratos calcula el vector de la consulta…"):
                        query = _embed_cached(client, [query_item], instruction)[0]
                    st.session_state["comparador_query_result"] = {
                        "signature": query_signature, "vector": query,
                    }
                except (GPUClientError, OSError, ValueError) as exc:
                    st.error(str(exc))
        saved = st.session_state.get("comparador_query_result")
        if saved and saved["signature"] == query_signature:
            query = saved["vector"]
        elif saved:
            st.info("La consulta cambió. Pulsa «Calcular nueva consulta» para actualizar vecinos.")

    count = len(items) - (1 if excluded is not None else 0)
    top_k = st.slider("Vecinos a mostrar", 1, min(20, count), min(5, count),
                      key="comparador_top_k")
    if query is not None:
        _render_neighbors(items, matrix, query, excluded, top_k)
    with st.expander("Qué observar"):
        st.write("Prueba A como texto, B como imagen y C como texto + imagen. Añade imágenes de una misma familia, calcula el mapa y consulta cuál queda más cerca. Cambia solo el texto de una ranura mixta para observar cómo se mueve su embedding y cómo cambian los vecinos reales.")


__all__ = ["render_comparator"]
