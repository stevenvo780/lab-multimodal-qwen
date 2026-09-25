"""Laboratorio Streamlit local para explorar embeddings multimodales en kratos.

Inicio: streamlit run laboratorio.py --server.address 127.0.0.1 --server.port 8501
La inferencia ocurre exclusivamente en la API GPU accesible por el túnel SSH.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from sklearn.decomposition import PCA

from gpu_client import GPUClient, GPUClientError, encode_image


ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "samples"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
MAX_IMAGE_BYTES = 8 * 1024 * 1024
DEFAULT_QUERIES = "un perro en la playa\nun gato\nun avión\nuna pizza"
DEFAULT_DOCUMENTS = (
    "un perro en la playa\nun gato descansando\nun avión volando\n"
    "una pizza\nla Torre Eiffel\nuna bicicleta\nun perro en la montaña"
)
RETRIEVAL_INSTRUCTION = "Retrieve images or text relevant to the user's query."
ALTERNATIVE_INSTRUCTION = "Find images that best match the user's description."


@dataclass(frozen=True)
class LabItem:
    label: str
    modality: str
    text: str = ""
    image: bytes = b""

    def api_item(self) -> dict[str, str]:
        result: dict[str, str] = {}
        if self.text.strip():
            result["text"] = self.text.strip()
        if self.image:
            result["image_base64"] = encode_image(self.image)
        return result

    def key(self) -> dict[str, str]:
        return {
            "label": self.label,
            "modality": self.modality,
            "text": self.text,
            "image_sha256": hashlib.sha256(self.image).hexdigest() if self.image else "",
        }


def fingerprint(*parts: Any) -> str:
    def convert(value: Any) -> Any:
        if isinstance(value, LabItem):
            return value.key()
        if isinstance(value, (list, tuple)):
            return [convert(part) for part in value]
        if isinstance(value, dict):
            return {key: convert(part) for key, part in value.items()}
        return value

    packed = json.dumps(convert(parts), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()


def lines(text: str, *, limit: int = 30) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()][:limit]


def text_items(text: str, *, prefix: str = "Texto") -> list[LabItem]:
    return [LabItem(f"{prefix} {index}: {value}", "texto", text=value)
            for index, value in enumerate(lines(text), start=1)]


def sample_files() -> list[Path]:
    if not SAMPLES.is_dir():
        return []
    return sorted(path for path in SAMPLES.iterdir()
                  if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def load_image(label: str, data: bytes) -> LabItem | None:
    if not data or len(data) > MAX_IMAGE_BYTES:
        st.warning(f"{label}: archivo vacío o de más de 8 MiB; se omite.")
        return None
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
    except (OSError, ValueError):
        st.warning(f"{label}: el archivo no es una imagen válida; se omite.")
        return None
    return LabItem(label=label, modality="imagen", image=data)


@st.cache_data(ttl=5, show_spinner=False)
def remote_health(base_url: str) -> dict[str, Any]:
    return GPUClient(base_url).health()


def normalized_matrix(vectors: list[list[float]]) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != 2048:
        raise GPUClientError("Los embeddings tienen forma inesperada.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise GPUClientError("La GPU devolvió un vector sin longitud útil.")
    return matrix / norms


def embed_items(client: GPUClient, items: list[LabItem], instruction: str) -> np.ndarray:
    payload = [item.api_item() for item in items]
    return normalized_matrix(client.embed(payload, instruction=instruction or None))


def heatmap(
    values: np.ndarray,
    rows: list[str],
    columns: list[str],
    *,
    title: str,
    difference: bool = False,
) -> go.Figure:
    if difference:
        bound = max(0.01, float(np.abs(values).max()))
        colorscale, low, high = "RdBu", -bound, bound
    else:
        colorscale, low, high = "Viridis", -1.0, 1.0
    figure = go.Figure(go.Heatmap(
        z=values,
        x=columns,
        y=rows,
        zmin=low,
        zmax=high,
        colorscale=colorscale,
        reversescale=difference,
        text=np.round(values, 2),
        texttemplate="%{text:.2f}",
        hovertemplate="%{y}<br>%{x}<br>valor: %{z:.3f}<extra></extra>",
        colorbar={"title": "Δ coseno" if difference else "coseno"},
    ))
    figure.update_layout(
        title=title,
        height=max(360, 90 + 58 * len(rows)),
        margin={"l": 30, "r": 20, "t": 65, "b": 90},
        xaxis_tickangle=-30,
        yaxis_autorange="reversed",
    )
    return figure


def ranked_results(scores: np.ndarray, items: list[LabItem], top_k: int) -> list[tuple[LabItem, float]]:
    order = np.argsort(scores)[::-1][:top_k]
    return [(items[int(index)], float(scores[int(index)])) for index in order]


def three_vector_geometry(vectors: np.ndarray) -> tuple[np.ndarray, list[dict[str, float | str]]]:
    """Encaja exactamente tres vectores unitarios en 3D a partir de su matriz de Gram."""
    triad = normalized_matrix(vectors.tolist())
    gram = np.clip(triad @ triad.T, -1.0, 1.0)
    gram = (gram + gram.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    coordinates = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0)))
    angles = []
    for first, second in ((0, 1), (1, 2), (0, 2)):
        cosine = float(np.clip(gram[first, second], -1, 1))
        angles.append({
            "Par": f"{'ABC'[first]}–{'ABC'[second]}",
            "Ángulo (°)": round(float(np.degrees(np.arccos(cosine))), 1),
            "Coseno": round(cosine, 3),
        })
    return coordinates, angles


def three_vector_figure(vectors: np.ndarray, show_result: bool) -> go.Figure:
    coordinates, _ = three_vector_geometry(vectors)
    figure = go.Figure()
    palette = {"A": "#41c7a6", "B": "#f59e0b", "C": "#9b8cff", "R": "#f472b6"}
    points = [(letter, coordinates[index]) for index, letter in enumerate("ABC")]
    if show_result:
        result = coordinates[0] - coordinates[1] + coordinates[2]
        result /= np.linalg.norm(result) + 1e-12
        points.append(("R", result))
    for label, point in points:
        figure.add_trace(go.Scatter3d(
            x=[0, float(point[0])], y=[0, float(point[1])], z=[0, float(point[2])],
            mode="lines+markers+text", name=label,
            line={"color": palette[label], "width": 8 if label != "R" else 5,
                  "dash": "solid" if label != "R" else "dash"},
            marker={"color": palette[label], "size": [2, 7]},
            text=["", label], textposition="top center",
            hovertemplate=f"{label}<br>x=%{{x:.2f}}, y=%{{y:.2f}}, z=%{{z:.2f}}<extra></extra>",
        ))
    figure.update_layout(
        title="A, B y C: sus tres ángulos reales",
        height=510,
        scene={
            "xaxis": {"range": [-1.1, 1.1], "title": "Eje 1"},
            "yaxis": {"range": [-1.1, 1.1], "title": "Eje 2"},
            "zaxis": {"range": [-1.1, 1.1], "title": "Eje 3"},
            "aspectmode": "cube",
        },
        margin={"l": 0, "r": 0, "t": 55, "b": 0},
    )
    return figure


def show_ranked(results: list[tuple[LabItem, float]]) -> None:
    for rank, (item, score) in enumerate(results, start=1):
        left, right = st.columns([1, 4], vertical_alignment="center")
        with left:
            if item.image:
                st.image(item.image, width=135)
            else:
                st.markdown("📝")
        with right:
            st.markdown(f"**{rank}. {item.label}**")
            st.caption(f"Similitud coseno: {score:.3f} · {item.modality}")


def show_stale_result(name: str, current_key: str) -> dict[str, Any] | None:
    result = st.session_state.get(name)
    if not result:
        return None
    if result["key"] != current_key:
        st.info("Las entradas cambiaron. Pulsa «Calcular» para ver resultados con estos ajustes.")
        return None
    return result


def count_slider(label: str, count: int, key: str) -> int:
    """Evita sliders de Streamlit con límites iguales al vaciar la colección."""
    if count < 2:
        st.caption(f"{label}: {count} elemento disponible.")
        return 1
    return st.slider(label, 1, count, min(5, count), key=key)


def matrix_tab(client: GPUClient, images: list[LabItem], query_instruction: str) -> None:
    st.subheader("1 · Matriz texto ↔ imagen")
    st.write("Cada celda compara una frase con una imagen en el mismo espacio vectorial.")
    query_text = st.text_area("Una consulta por línea", DEFAULT_QUERIES, height=130, key="matrix_queries")
    compare = st.checkbox("Comparar con otra instrucción", key="matrix_compare")
    alternative = ""
    if compare:
        alternative = st.text_input(
            "Segunda instrucción (en inglés)",
            ALTERNATIVE_INSTRUCTION,
            key="matrix_alternative",
        )
        st.caption("Cada instrucción se aplica solo a las consultas. Las imágenes se codifican sin instrucción de búsqueda.")
    queries = text_items(query_text, prefix="Consulta")
    key = fingerprint(queries, images, query_instruction, compare, alternative)
    if st.button("Calcular matriz", type="primary", key="matrix_run"):
        if not queries or not images:
            st.error("Escribe una frase y selecciona al menos una imagen.")
        else:
            try:
                with st.spinner("Kratos está calculando los embeddings…"):
                    image_vectors = embed_items(client, images, "")
                    query_vectors = embed_items(client, queries, query_instruction)
                    scores = query_vectors @ image_vectors.T
                    comparison = None
                    if compare:
                        if not alternative.strip():
                            raise GPUClientError("Escribe la segunda instrucción para compararlas.")
                        other_queries = embed_items(client, queries, alternative)
                        comparison = other_queries @ image_vectors.T
                    st.session_state["matrix_result"] = {
                        "key": key, "scores": scores, "comparison": comparison,
                    }
            except GPUClientError as exc:
                st.error(str(exc))
    result = show_stale_result("matrix_result", key)
    if result:
        row_names = [item.text for item in queries]
        column_names = [item.label for item in images]
        st.plotly_chart(heatmap(result["scores"], row_names, column_names,
                                title="Similitud coseno"), width="stretch")
        best = np.argmax(result["scores"], axis=1)
        st.dataframe([{
            "Consulta": query.text,
            "Imagen más cercana": images[int(best[index])].label,
            "Coseno": round(float(result["scores"][index, best[index]]), 3),
        } for index, query in enumerate(queries)], hide_index=True, width="stretch")
        if result["comparison"] is not None:
            delta = result["comparison"] - result["scores"]
            st.plotly_chart(heatmap(delta, row_names, column_names,
                                    title="Cambio al usar otra instrucción", difference=True),
                            width="stretch")
            st.caption("Un cambio positivo significa mayor coseno para ese par; no garantiza mejor búsqueda.")
    with st.expander("Qué observar"):
        st.write("Busca una fila donde la imagen correcta no quede primera. Cambia una palabra de la consulta o la instrucción y compara las celdas. El coseno es una medida relativa, no una probabilidad.")


def search_tab(client: GPUClient, images: list[LabItem], documents: list[LabItem], query_instruction: str) -> None:
    st.subheader("2 · Búsqueda multimodal")
    st.write("La consulta y los candidatos se codifican por separado; después se ordenan por coseno.")
    query_text = st.text_input("Consulta", "un perro en la playa", key="search_query")
    image_options = ["Sin imagen"] + [item.label for item in images]
    query_image_name = st.selectbox("Imagen en la consulta (opcional)", image_options, key="search_image")
    add_filename = st.checkbox(
        "Combinar cada imagen con el texto de su nombre de archivo",
        value=False, key="search_filename",
        help="Permite comparar imagen sola frente a texto + imagen. El nombre pasa a formar parte de la entrada.",
    )
    exclude_self = st.checkbox("Excluir la misma imagen de los resultados", value=True, key="search_exclude")
    query_image = next((item for item in images if item.label == query_image_name), None)
    query = LabItem("Consulta", "mixto" if query_image and query_text.strip() else
                    "imagen" if query_image else "texto",
                    text=query_text.strip(), image=query_image.image if query_image else b""
                    )
    candidates = [
        LabItem(item.label, "mixto" if add_filename else "imagen",
                text=Path(item.label).stem.replace("-", " ") if add_filename else "",
                image=item.image)
        for item in images
        if not (exclude_self and query_image and item.label == query_image.label)
    ] + documents
    top_k = count_slider("Resultados a mostrar", len(candidates), "search_top_k")
    key = fingerprint(query, candidates, query_instruction, top_k)
    if st.button("Buscar", type="primary", key="search_run"):
        if not query.api_item():
            st.error("Escribe texto o elige una imagen para la consulta.")
        elif not candidates:
            st.error("Añade alguna imagen o texto a la colección.")
        else:
            try:
                with st.spinner("Kratos está comparando la consulta con la colección…"):
                    query_vector = embed_items(client, [query], query_instruction)[0]
                    candidate_vectors = embed_items(client, candidates, "")
                    scores = candidate_vectors @ query_vector
                    st.session_state["search_result"] = {
                        "key": key, "ranked": ranked_results(scores, candidates, top_k),
                    }
            except GPUClientError as exc:
                st.error(str(exc))
    result = show_stale_result("search_result", key)
    if result:
        show_ranked(result["ranked"])
    with st.expander("Qué observar"):
        st.write("Cambia la frase sin cambiar la colección. Prueba una consulta solo con imagen y otra con imagen + texto. Si buscas con una imagen que está en la colección, excluirla evita que aparezca como primer resultado trivial.")


def arithmetic_input(letter: str, images: list[LabItem], default_text: str) -> LabItem:
    with st.container(border=True):
        st.markdown(f"**{letter}**")
        phrase = st.text_input("Texto (opcional)", default_text, key=f"arithmetic_{letter}_text")
        options = ["Sin imagen"] + [image.label for image in images]
        image_name = st.selectbox("Imagen (opcional)", options, key=f"arithmetic_{letter}_image")
    selected = next((image for image in images if image.label == image_name), None)
    modality = "mixto" if selected and phrase.strip() else "imagen" if selected else "texto"
    return LabItem(letter, modality, text=phrase.strip(), image=selected.image if selected else b""
                   )


def arithmetic_tab(client: GPUClient, images: list[LabItem], documents: list[LabItem]) -> None:
    st.subheader("3 · A − B + C")
    st.write("Suma y resta vectores normalizados; busca qué candidato queda más cerca del vector resultante.")
    st.caption("Aquí todas las entradas usan la instrucción predeterminada del encoder.")
    cols = st.columns(3)
    with cols[0]:
        a = arithmetic_input("A", images, "un perro en la playa")
    with cols[1]:
        b = arithmetic_input("B", images, "playa")
    with cols[2]:
        c = arithmetic_input("C", images, "montaña")
    candidates = images + documents
    top_k = count_slider("Candidatos a mostrar", len(candidates), "arithmetic_top_k")
    key = fingerprint(a, b, c, candidates, top_k)
    if st.button("Probar aritmética", type="primary", key="arithmetic_run"):
        if any(not item.api_item() for item in (a, b, c)):
            st.error("A, B y C necesitan texto, imagen o ambos.")
        elif not candidates:
            st.error("Añade candidatos a la colección.")
        else:
            try:
                with st.spinner("Kratos está codificando A, B, C y los candidatos…"):
                    vectors = embed_items(client, [a, b, c] + candidates, "")
                    composite = vectors[0] - vectors[1] + vectors[2]
                    magnitude = float(np.linalg.norm(composite))
                    if magnitude < 1e-8:
                        raise GPUClientError("A − B + C produjo un vector casi nulo; cambia las entradas.")
                    composite /= magnitude
                    scores = vectors[3:] @ composite
                    st.session_state["arithmetic_result"] = {
                        "key": key,
                        "ranked": ranked_results(scores, candidates, top_k),
                        "magnitude": magnitude,
                        "ab": float(vectors[0] @ vectors[1]),
                        "ac": float(vectors[0] @ vectors[2]),
                        "triad": vectors[:3].copy(),
                    }
            except GPUClientError as exc:
                st.error(str(exc))
    result = show_stale_result("arithmetic_result", key)
    if result:
        st.code("resultado = normalizar(embedding(A) − embedding(B) + embedding(C))", language="python")
        metric_cols = st.columns(3)
        metric_cols[0].metric("Coseno A ↔ B", f"{result['ab']:.3f}")
        metric_cols[1].metric("Coseno A ↔ C", f"{result['ac']:.3f}")
        metric_cols[2].metric("Longitud antes de normalizar", f"{result['magnitude']:.3f}")
        st.markdown("#### Tres vectores, tres ángulos")
        st.caption("Los tres vectores de 2048 dimensiones se colocan en 3D conservando exactamente sus tres ángulos. Puedes girar el gráfico. Los ejes 3D no tienen significado semántico propio.")
        show_result = st.checkbox("Mostrar también R = normalizar(A − B + C)", value=False,
                                  key="arithmetic_show_result")
        diagram, table = st.columns([2, 1])
        with diagram:
            st.plotly_chart(three_vector_figure(result["triad"], show_result),
                            width="stretch")
        with table:
            _, angles = three_vector_geometry(result["triad"])
            st.dataframe(angles, hide_index=True, width="stretch")
            st.info("Un ángulo menor corresponde a mayor coseno y más cercanía según este modelo. Los ángulos son relaciones entre vectores; no son probabilidades.")
        st.markdown("#### Vecinos del resultado")
        show_ranked(result["ranked"])
    with st.expander("Qué observar"):
        st.write("La aritmética puede acercarse a una analogía, pero no hay garantía de que separe conceptos de forma limpia. Prueba cambiar solo B o C y mira si el primer candidato cambia como esperabas.")


def scatter_figure(points: np.ndarray, items: list[LabItem], title: str) -> go.Figure:
    figure = go.Figure()
    palette = {"texto": "#2d8af0", "imagen": "#f97316", "mixto": "#a855f7"}
    for modality in ("texto", "imagen", "mixto"):
        indexes = [index for index, item in enumerate(items) if item.modality == modality]
        if not indexes:
            continue
        labels = [items[index].label for index in indexes]
        figure.add_trace(go.Scatter(
            x=points[indexes, 0], y=points[indexes, 1],
            mode="markers+text", name=modality,
            text=[label if len(label) <= 19 else label[:16] + "…" for label in labels],
            textposition="top center",
            marker={"size": 13, "color": palette[modality], "line": {"width": 1, "color": "white"}},
            customdata=labels,
            hovertemplate="%{customdata}<br>x=%{x:.3f}, y=%{y:.3f}<extra></extra>",
        ))
    figure.update_layout(
        title=title, height=490, hovermode="closest",
        margin={"l": 30, "r": 20, "t": 65, "b": 45},
        xaxis_title="Componente 1", yaxis_title="Componente 2",
        legend={"orientation": "h", "y": -0.18},
    )
    return figure


def projection_tab(client: GPUClient, images: list[LabItem], documents: list[LabItem]) -> None:
    st.subheader("4 · Mapa 2D: PCA y UMAP")
    st.write("Reduce los vectores de 2048 dimensiones a dos coordenadas para explorar grupos.")
    st.caption("Todos los puntos usan la instrucción predeterminada del encoder.")
    items = images + documents
    max_neighbors = max(2, len(items) - 1)
    if max_neighbors == 2:
        neighbors = 2
        st.caption("Vecinos de UMAP: 2 (amplía la colección para ajustarlos).")
    else:
        neighbors = st.slider("Vecinos de UMAP", 2, max_neighbors,
                              min(8, max_neighbors), key="projection_neighbors")
    min_dist = st.slider("Separación mínima de UMAP", 0.0, 0.9, 0.15, 0.05,
                         key="projection_min_dist")
    key = fingerprint(items, neighbors, min_dist)
    if st.button("Proyectar colección", type="primary", key="projection_run"):
        if len(items) < 4:
            st.error("Selecciona al menos cuatro elementos entre imágenes y textos.")
        else:
            try:
                with st.spinner("Kratos calcula los vectores; después PCA y UMAP corren en tu equipo…"):
                    vectors = embed_items(client, items, "")
                    pca = PCA(n_components=2)
                    pca_points = pca.fit_transform(vectors)
                    umap_points = None
                    umap_error = ""
                    try:
                        import umap

                        umap_points = umap.UMAP(
                            n_components=2,
                            n_neighbors=neighbors,
                            min_dist=min_dist,
                            metric="cosine",
                            init="random",
                            random_state=42,
                        ).fit_transform(vectors)
                    except (ImportError, ValueError, RuntimeError) as exc:
                        umap_error = str(exc)
                    similarity = vectors @ vectors.T
                    np.fill_diagonal(similarity, -np.inf)
                    first, second = np.unravel_index(int(np.argmax(similarity)), similarity.shape)
                    st.session_state["projection_result"] = {
                        "key": key, "pca_points": pca_points,
                        "pca_variance": float(pca.explained_variance_ratio_.sum()),
                        "umap_points": umap_points, "umap_error": umap_error,
                        "nearest": (items[int(first)].label, items[int(second)].label,
                                    float(similarity[first, second])),
                    }
            except GPUClientError as exc:
                st.error(str(exc))
    result = show_stale_result("projection_result", key)
    if result:
        st.caption(f"PCA muestra {result['pca_variance']:.1%} de la varianza de estos datos en 2D.")
        col_pca, col_umap = st.columns(2)
        with col_pca:
            st.plotly_chart(scatter_figure(result["pca_points"], items, "PCA"),
                            width="stretch")
        with col_umap:
            if result["umap_points"] is not None:
                st.plotly_chart(scatter_figure(result["umap_points"], items, "UMAP"),
                                width="stretch")
            else:
                st.warning(f"UMAP no pudo calcularse: {result['umap_error']}")
        first, second, score = result["nearest"]
        st.info(f"En las 2048 dimensiones, el par más cercano es «{first}» ↔ «{second}» (coseno {score:.3f}).")
    with st.expander("Qué observar"):
        st.write("Compara vecinos en los dos gráficos y luego usa el coseno original. PCA conserva varianza lineal; UMAP enfatiza vecindades. Las distancias y tamaños de grupos del mapa 2D pueden deformarse.")


def main() -> None:
    embedded_shell = st.query_params.get("shell") == "1"
    st.set_page_config(page_title="Laboratorio multimodal Qwen", page_icon="🧭", layout="wide",
                       initial_sidebar_state="collapsed" if embedded_shell else "expanded")
    if embedded_shell:
        st.markdown("""
        <style>
        [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
        [data-testid="stExpandSidebarButton"] {
            display: none !important;
        }
        .block-container { max-width: 100rem; padding-top: 1.5rem; }
        </style>
        """, unsafe_allow_html=True)
    client = GPUClient()
    module = st.query_params.get("modulo", "")
    if module == "tomografia":
        from tomografia_lab import render_tomography

        with st.sidebar:
            st.header("Recorrido")
            st.markdown("[← Cuatro experimentos](http://127.0.0.1:8501/)")
            st.markdown("[A, B y C · ángulos reales](http://127.0.0.1:8501/?modulo=aritmetica)")
            try:
                remote_health(client.base_url)
                st.success("GPU de kratos conectada")
            except GPUClientError as exc:
                st.error(str(exc))
        render_tomography(client)
        return

    if module in {"caso_empresa", "comparador"}:
        with st.sidebar:
            st.header("Recorrido")
            st.markdown("[← Cuatro experimentos](http://127.0.0.1:8501/)")
            st.markdown("[Comparador multimodal A/B/C](http://127.0.0.1:8501/?modulo=comparador)")
            st.markdown("[Caso de documentos](http://127.0.0.1:8501/?modulo=caso_empresa)")
            st.markdown("[Tomografía de capas](http://127.0.0.1:8501/?modulo=tomografia)")
            try:
                remote_health(client.base_url)
                st.success("GPU de kratos conectada")
            except GPUClientError as exc:
                st.error(str(exc))
        if module == "caso_empresa":
            from caso_empresa import render_enterprise_case

            render_enterprise_case(client)
        else:
            from comparador_lab import render_comparator

            render_comparator(client)
        return

    st.title("Laboratorio de embeddings multimodales")
    st.caption("Texto + imágenes → Qwen3-VL-Embedding-2B en kratos → vectores → experimentos locales")
    st.markdown("[Tomografía: capas, entrenamiento y predicción →](http://127.0.0.1:8501/?modulo=tomografia)")
    st.markdown("[Comparador 3D: A/B/C con tus imágenes y textos →](http://127.0.0.1:8501/?modulo=comparador)")
    st.markdown("[Caso de documentos: árbol, espacio 3D y clasificación →](http://127.0.0.1:8501/?modulo=caso_empresa)")

    with st.sidebar:
        st.header("Conexión")
        if st.button("Comprobar de nuevo", key="health_refresh"):
            remote_health.clear()
        try:
            health = remote_health(client.base_url)
            if health["model_loaded"]:
                st.success("Kratos listo · modelo cargado · 2048 dimensiones")
            else:
                st.warning("Servicio conectado; el modelo aún está cargando.")
        except GPUClientError as exc:
            st.error(str(exc))
        st.caption(f"API: {client.base_url}")
        st.divider()
        st.header("Instrucción de búsqueda")
        query_instruction = st.text_input(
            "Para consultas de matriz y búsqueda",
            value=RETRIEVAL_INSTRUCTION,
            placeholder="Vacío = instrucción predeterminada",
            help="Se aplica a las consultas. Las imágenes y los textos candidatos usan la instrucción predeterminada del modelo.",
        )
        st.caption("La [receta oficial](https://github.com/QwenLM/Qwen3-VL-Embedding#usage) instruye la consulta y codifica documentos sin esa instrucción.")

    st.markdown("### Colección para experimentar")
    st.write("Selecciona imágenes de ejemplo y añade frases. Estos elementos serán los candidatos de búsqueda, aritmética y proyección.")
    available = sample_files()
    names = [path.name for path in available]
    preferred = [name for name in (
        "perro.jpg", "gato.jpg", "playa-perro.jpg", "avion.jpg",
        "pizza.jpg", "bicicleta.jpg", "torre-eiffel.jpg", "everest.jpg",
    ) if name in names]
    chosen = st.multiselect("Imágenes de samples/", names,
                            default=preferred or names[:6], key="selected_samples")
    uploaded = st.file_uploader("También puedes subir imágenes", type=["jpg", "jpeg", "png", "webp"],
                                accept_multiple_files=True, key="uploaded_samples")
    images: list[LabItem] = []
    sample_map = {path.name: path for path in available}
    for name in chosen:
        try:
            item = load_image(name, sample_map[name].read_bytes())
            if item:
                images.append(item)
        except OSError as exc:
            st.warning(f"No se pudo leer {name}: {exc}")
    for upload in uploaded or []:
        item = load_image(f"subido: {upload.name}", upload.getvalue())
        if item:
            images.append(item)
    document_text = st.text_area(
        "Textos candidatos (uno por línea)", DEFAULT_DOCUMENTS,
        height=155, key="candidate_texts",
    )
    documents = text_items(document_text)
    st.caption(f"Colección actual: {len(images)} imágenes y {len(documents)} textos. Máximo 30 líneas de texto y 8 MiB por imagen.")
    if images:
        with st.expander("Ver imágenes seleccionadas"):
            for offset in range(0, len(images), 4):
                cols = st.columns(4)
                for col, item in zip(cols, images[offset:offset + 4]):
                    with col:
                        st.image(item.image, caption=item.label, width="stretch")

    st.divider()
    if module == "aritmetica":
        st.markdown("[← Volver a los cuatro experimentos](http://127.0.0.1:8501/)")
        arithmetic_tab(client, images, documents)
    else:
        matrix, search, arithmetic, projection = st.tabs([
            "Matriz texto ↔ imagen", "Búsqueda", "A − B + C", "PCA / UMAP",
        ])
        with matrix:
            matrix_tab(client, images, query_instruction)
        with search:
            search_tab(client, images, documents, query_instruction)
        with arithmetic:
            arithmetic_tab(client, images, documents)
        with projection:
            projection_tab(client, images, documents)

    st.divider()
    st.caption("Los gráficos y la aritmética se calculan en este equipo; los embeddings salen de kratos. Fuentes: [Qwen3-VL-Embedding](https://github.com/QwenLM/Qwen3-VL-Embedding) · [Ficha del modelo](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B).")


if __name__ == "__main__":
    main()
