"""Caso didáctico de documentos: árbol de vecinos, mapa 3D y clasificación.

Las distancias de referencia siempre se calculan en los 2048 valores originales.
La proyección 3D es una ayuda visual, no otra métrica del modelo.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageOps
from sklearn.decomposition import PCA

from gpu_client import GPUClient, GPUClientError, encode_image


ROOT = Path(__file__).resolve().parent
CASE = ROOT / "data" / "empresa"
MANIFEST = CASE / "manifest.json"
EMBEDDINGS = CASE / "embeddings.json"

TYPES = {
    "bank_statement": "Extracto bancario",
    "chart": "Gráfico",
    "financial_table": "Tabla financiera",
    "lease": "Contrato de arriendo",
    "medical": "Documento médico",
    "payslip": "Nómina",
    "photo_receipt": "Recibo fotografiado",
    "receipt": "Recibo escaneado",
    "shipping": "Facturación de envíos",
    "tax_form": "Formulario fiscal",
}
PALETTE = [
    "#2aa889", "#ed9b42", "#8876d9", "#4ca6d8", "#d55c8c",
    "#5dba68", "#d2bc43", "#e17160", "#6685d8", "#997457",
]
COLORS = {name: PALETTE[index] for index, name in enumerate(sorted(TYPES))}


def display_type(label: str) -> str:
    return TYPES.get(label, label.replace("_", " "))


def unit_matrix(vectors: list[list[float]]) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != 2048 or not np.isfinite(matrix).all():
        raise ValueError("Los embeddings guardados no tienen 2048 valores finitos.")
    lengths = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(lengths < 1e-8):
        raise ValueError("Hay un embedding de longitud cero.")
    return matrix / lengths


@st.cache_data(show_spinner=False)
def read_case(manifest_stamp: float, embeddings_stamp: float) -> tuple[list[dict], np.ndarray, np.ndarray]:
    del manifest_stamp, embeddings_stamp
    documents = json.loads(MANIFEST.read_text(encoding="utf-8"))["documents"]
    stored = json.loads(EMBEDDINGS.read_text(encoding="utf-8"))
    missing = [doc["id"] for doc in documents if not {"image", "text"} <= stored.get(doc["id"], {}).keys()]
    if missing:
        raise ValueError(f"Faltan embeddings para {len(missing)} documentos.")
    images = unit_matrix([stored[doc["id"]]["image"] for doc in documents])
    texts = unit_matrix([stored[doc["id"]]["text"] for doc in documents])
    return documents, images, texts


def image_bytes(document: dict) -> bytes:
    return (CASE / document["image"]).read_bytes()


def web_image(raw: bytes, *, longest: int = 768) -> bytes:
    """Valida y reduce imágenes añadidas al tamaño de la muestra."""
    if len(raw) > 16 * 1024 * 1024:
        raise ValueError("La imagen supera 16 MiB.")
    with Image.open(io.BytesIO(raw)) as source:
        if source.width * source.height > 30_000_000:
            raise ValueError("La imagen tiene demasiados píxeles.")
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((longest, longest), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        image.save(output, "JPEG", quality=82, optimize=True)
    return output.getvalue()


def neighbor_tree(images: np.ndarray, root: int) -> list[dict]:
    """Expande tres vecinos del origen y dos de cada uno, sin repetir páginas."""
    scores = images @ images.T
    taken = {root}
    nodes = [{"index": root, "parent": None, "score": None, "depth": 0, "y": 0.0}]
    first = []
    for y in (-3.0, 0.0, 3.0):
        candidates = [int(index) for index in np.argsort(scores[root])[::-1] if int(index) not in taken]
        if not candidates:
            break
        index = candidates[0]
        taken.add(index)
        node = {"index": index, "parent": root, "score": float(scores[root, index]),
                "depth": 1, "y": y}
        nodes.append(node)
        first.append(node)
    for parent in first:
        parent_index = parent["index"]
        for offset in (-0.65, 0.65):
            candidates = [int(index) for index in np.argsort(scores[parent_index])[::-1]
                          if int(index) not in taken]
            if not candidates:
                break
            index = candidates[0]
            taken.add(index)
            nodes.append({"index": index, "parent": parent_index,
                          "score": float(scores[parent_index, index]),
                          "depth": 2, "y": parent["y"] + offset})
    return nodes


def tree_figure(documents: list[dict], nodes: list[dict]) -> go.Figure:
    figure = go.Figure()
    by_index = {node["index"]: node for node in nodes}
    for node in nodes[1:]:
        parent = by_index[node["parent"]]
        figure.add_trace(go.Scatter(
            x=[parent["depth"], node["depth"]], y=[parent["y"], node["y"]],
            mode="lines", line={"color": "#a7b4c3", "width": 2},
            hoverinfo="skip", showlegend=False,
        ))
        figure.add_annotation(
            x=(parent["depth"] + node["depth"]) / 2,
            y=(parent["y"] + node["y"]) / 2,
            text=f"{node['score']:.2f}", showarrow=False,
            bgcolor="rgba(255,255,255,0.88)", font={"size": 11, "color": "#435467"},
        )
    for label in sorted({documents[node["index"]]["label"] for node in nodes}):
        selected = [node for node in nodes if documents[node["index"]]["label"] == label]
        figure.add_trace(go.Scatter(
            x=[node["depth"] for node in selected], y=[node["y"] for node in selected],
            mode="markers+text", name=display_type(label),
            text=[documents[node["index"]]["id"].replace("_", " #") for node in selected],
            textposition="middle right", textfont={"size": 11},
            marker={"size": [24 if node["depth"] == 0 else 17 for node in selected],
                    "color": COLORS[label], "line": {"color": "white", "width": 2}},
            customdata=[
                [documents[node["index"]]["id"], display_type(label),
                 "origen" if node["score"] is None else f"coseno {node['score']:.3f}"]
                for node in selected
            ],
            hovertemplate="%{customdata[0]}<br>%{customdata[1]}<br>%{customdata[2]}<extra></extra>",
        ))
    figure.update_layout(
        height=550, margin={"l": 30, "r": 115, "t": 25, "b": 35},
        xaxis={"tickvals": [0, 1, 2], "ticktext": ["Documento elegido", "Vecinos", "Vecinos de vecinos"],
               "range": [-0.15, 2.7], "showgrid": False, "zeroline": False},
        yaxis={"visible": False, "range": [-4.3, 4.3]},
        legend={"orientation": "h", "y": -0.16},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return figure


@st.cache_data(show_spinner=False)
def projected_3d(images: np.ndarray, texts: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    n = len(images)
    pca = PCA(n_components=3)
    points = pca.fit_transform(np.vstack((images, texts)))
    return points[:n], points[n:], float(pca.explained_variance_ratio_.sum())


def map_figure(documents: list[dict], images: np.ndarray, texts: np.ndarray,
               selected: int, show_text: bool) -> tuple[go.Figure, float]:
    image_points, text_points, variance = projected_3d(images, texts)
    figure = go.Figure()
    for label in sorted(TYPES):
        indexes = [index for index, doc in enumerate(documents) if doc["label"] == label]
        if not indexes:
            continue
        for modality, points, symbol in (("imagen", image_points, "circle"),
                                         ("transcripción", text_points, "diamond")):
            if modality == "transcripción" and not show_text:
                continue
            figure.add_trace(go.Scatter3d(
                x=points[indexes, 0], y=points[indexes, 1], z=points[indexes, 2],
                mode="markers", name=f"{display_type(label)} · {modality}",
                legendgroup=label, showlegend=modality == "imagen",
                marker={"size": [12 if index == selected else 6 for index in indexes],
                        "symbol": symbol, "color": COLORS[label],
                        "opacity": 0.95 if modality == "imagen" else 0.55,
                        "line": {"color": "white", "width": 1}},
                customdata=[[documents[index]["id"], modality, display_type(label)] for index in indexes],
                hovertemplate="%{customdata[0]}<br>%{customdata[2]}<br>%{customdata[1]}<extra></extra>",
            ))
    if show_text:
        first, second = image_points[selected], text_points[selected]
        figure.add_trace(go.Scatter3d(
            x=[first[0], second[0]], y=[first[1], second[1]], z=[first[2], second[2]],
            mode="lines", name="Imagen ↔ transcripción seleccionada", showlegend=False,
            line={"color": "#243c55", "width": 6, "dash": "dash"},
            hoverinfo="skip",
        ))
    figure.update_layout(
        height=620, margin={"l": 0, "r": 0, "t": 10, "b": 0},
        scene={"xaxis_title": "PCA 1", "yaxis_title": "PCA 2", "zaxis_title": "PCA 3",
               "aspectmode": "cube"},
        legend={"x": 0, "y": 1, "font": {"size": 10}},
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return figure, variance


def leave_one_out(images: np.ndarray, documents: list[dict]) -> list[dict]:
    labels = [doc["label"] for doc in documents]
    classes = sorted(set(labels))
    results = []
    for index, doc in enumerate(documents):
        centers = []
        for label in classes:
            peers = [images[position] for position, value in enumerate(labels)
                     if value == label and position != index]
            center = np.mean(peers, axis=0)
            center /= np.linalg.norm(center)
            centers.append(center)
        scores = np.asarray(centers) @ images[index]
        order = np.argsort(scores)[::-1]
        results.append({"id": doc["id"], "real": doc["label"],
                        "predicted": classes[int(order[0])],
                        "cosine": float(scores[order[0]]),
                        "margin": float(scores[order[0]] - scores[order[1]])})
    return results


def classify(vector: np.ndarray, images: np.ndarray, documents: list[dict]) -> list[tuple[str, float]]:
    classes = sorted({doc["label"] for doc in documents})
    labels = [doc["label"] for doc in documents]
    centers = []
    for label in classes:
        center = np.mean(images[[index for index, value in enumerate(labels) if value == label]], axis=0)
        centers.append(center / np.linalg.norm(center))
    scores = np.asarray(centers) @ vector
    return sorted(zip(classes, [float(score) for score in scores], strict=True),
                  key=lambda item: item[1], reverse=True)


def _classification_key(picture: bytes, client: GPUClient, images: np.ndarray,
                        documents: list[dict]) -> tuple[str, str]:
    """Identifica la imagen enviada y el contexto que determina su clasificación."""
    context = hashlib.sha256(json.dumps({
        "api_url": client.base_url,
        "labels": [doc["label"] for doc in documents],
        "shape": images.shape,
        "dtype": images.dtype.str,
    }, sort_keys=True).encode("utf-8"))
    context.update(images.tobytes())
    return hashlib.sha256(picture).hexdigest(), context.hexdigest()


def layer_chart(scans: dict) -> go.Figure:
    figure = go.Figure()
    for modality, scan in scans.items():
        layers = scan["layers"]
        figure.add_trace(go.Scatter(
            x=[layer["layer"] for layer in layers[1:]],
            y=[1 - layer["delta_cos"] for layer in layers[1:]],
            mode="lines+markers", name=modality,
            hovertemplate="Capa %{x}<br>cambio 1−cos: %{y:.4f}<extra></extra>",
        ))
    figure.update_layout(
        height=350, xaxis_title="Capa del decodificador", yaxis_title="Cambio 1 − coseno",
        margin={"l": 35, "r": 15, "t": 20, "b": 35},
        legend={"orientation": "h", "y": 1.18},
    )
    return figure


def render_enterprise_case(client: GPUClient) -> None:
    st.title("Documentos de empresa · laboratorio de relaciones")
    st.caption("51 páginas públicas, reales y sintéticas · Qwen3-VL-Embedding-2B · 2048 dimensiones · GPU de kratos")
    st.write("Elige un documento. Verás sus vecinos como árbol, sus representaciones de imagen y texto en 3D, y cómo una clasificación sencilla usa esa cercanía.")
    if not MANIFEST.exists() or not EMBEDDINGS.exists():
        st.info("Hay que preparar la muestra local y calcular los embeddings una vez.")
        if st.button("Preparar documentos y embeddings", type="primary"):
            try:
                from preparar_caso_empresa import embed, prepare
                with st.spinner("Preparando 51 documentos y calculando sus embeddings en kratos…"):
                    embed(prepare())
                st.rerun()
            except (OSError, ValueError, GPUClientError) as exc:
                st.error(str(exc))
        return
    try:
        documents, images, texts = read_case(MANIFEST.stat().st_mtime, EMBEDDINGS.stat().st_mtime)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        st.error(f"No se pudo cargar la muestra: {exc}")
        return
    kinds = Counter(doc["label"] for doc in documents)
    st.caption(f"{len(documents)} documentos · {len(kinds)} tipos · "
               "texto = transcripción de referencia, no OCR generado por Qwen")
    with st.expander("Procedencia de la muestra", expanded=False):
        st.markdown("Colección: [OCR Benchmark](https://huggingface.co/datasets/ilsilfverskiold/ocr-benchmark). "
                    "Usamos solo filas cuya fuente declara licencia MIT o CC BY 4.0: "
                    "[OmniAI](https://huggingface.co/datasets/getomni-ai/ocr-benchmark), "
                    "[SROIE](https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE) y "
                    "[SynFinTabs](https://huggingface.co/datasets/ethanbradley/synfintabs). "
                    "Incluye páginas sintéticas y páginas escaneadas reales. Es una demostración local pequeña; "
                    "las condiciones de las imágenes de terceros deben verificarse antes de reutilizarlas.")
    options = list(range(len(documents)))
    selected = st.selectbox(
        "Documento que quieres investigar", options,
        format_func=lambda index: f"{documents[index]['id']} · {display_type(documents[index]['label'])}",
        key="enterprise_selected",
    )
    document = documents[selected]
    preview, details = st.columns([1, 2], vertical_alignment="top")
    with preview:
        st.image(image_bytes(document), width="stretch")
    with details:
        st.markdown(f"### {document['id']}")
        st.write(f"**Tipo de referencia:** {display_type(document['label'])} (`{document['label']}`)")
        st.write(f"**Fuente:** [{document['source']}]({document['source_url']}) · licencia declarada {document['license']}")
        st.metric("Imagen ↔ su transcripción, coseno en 2048D", f"{images[selected] @ texts[selected]:.3f}")
        st.text_area("Transcripción de referencia (extracto)", document["text"][:900],
                     height=190, disabled=True)
    tree_tab, space_tab, classification_tab, inspect_tab = st.tabs([
        "Árbol de relaciones", "Espacio 3D", "Clasificar y buscar", "Tomografía",
    ])
    with tree_tab:
        st.subheader("De un documento a sus vecinos")
        st.write("Cada rama une un documento con uno de sus vecinos según el coseno de los embeddings de imagen en 2048D. "
                 "El árbol se construye al explorar; no es una estructura interna de Qwen.")
        nodes = neighbor_tree(images, selected)
        st.plotly_chart(tree_figure(documents, nodes), width="stretch")
        st.caption("Números en ramas: coseno en 2048D. Color: tipo conocido del dataset. "
                   "Una rama entre tipos distintos muestra cercanía visual aprendida, no que las etiquetas sean equivalentes.")
        st.dataframe([{
            "Documento": documents[node["index"]]["id"],
            "Tipo": display_type(documents[node["index"]]["label"]),
            "Nivel": node["depth"],
            "Coseno con su padre": round(node["score"], 3) if node["score"] is not None else None,
        } for node in nodes], hide_index=True, width="stretch")
    with space_tab:
        st.subheader("Imágenes y textos en un mismo espacio")
        show_text = st.checkbox("Mostrar también transcripciones", value=True, key="enterprise_show_text")
        figure, variance = map_figure(documents, images, texts, selected, show_text)
        st.plotly_chart(figure, width="stretch")
        st.caption(f"PCA condensa 2048D a 3D y conserva {variance:.1%} de la varianza de esta muestra. "
                   "Puedes girar y acercar el mapa; las distancias visibles pueden deformarse. "
                   "El segmento une la imagen seleccionada con su transcripción de referencia.")
        all_vectors = np.vstack((images, texts))
        similarities = all_vectors @ images[selected]
        similarities[selected] = -np.inf
        top = np.argsort(similarities)[::-1][:8]
        st.markdown("#### Vecinos exactos del documento elegido")
        st.dataframe([{
            "Documento": documents[int(index) % len(documents)]["id"],
            "Modalidad": "imagen" if index < len(documents) else "transcripción",
            "Tipo": display_type(documents[int(index) % len(documents)]["label"]),
            "Coseno 2048D": round(float(similarities[index]), 3),
        } for index in top], hide_index=True, width="stretch")
    with classification_tab:
        st.subheader("Aplicación: organizar documentos por tipo")
        st.write("Una regla sencilla compara cada imagen con el centro de las otras imágenes de cada tipo. "
                 "Para evaluar cada página, la excluimos de su propio centro (leave one out). Qwen permanece congelado.")
        outcomes = leave_one_out(images, documents)
        correct = sum(row["real"] == row["predicted"] for row in outcomes)
        st.metric("Coincidencias dentro de esta muestra", f"{correct}/{len(outcomes)}")
        st.warning("Hay cinco o seis páginas por tipo. Varias comparten plantillas y cada tipo "
                   "está asociado a una fuente concreta: el método puede reconocer estilo o procedencia. "
                   "Esta cuenta no estima calidad con documentos empresariales nuevos.")
        st.dataframe([{
            "Documento": row["id"], "Real": display_type(row["real"]),
            "Predicción": display_type(row["predicted"]),
            "Acierta": "sí" if row["real"] == row["predicted"] else "no",
            "Coseno": round(row["cosine"], 3), "Margen": round(row["margin"], 3),
        } for row in outcomes], hide_index=True, width="stretch")
        st.markdown("#### Prueba una imagen tuya")
        upload = st.file_uploader("Imagen de un documento", type=["jpg", "jpeg", "png", "webp"],
                                  key="enterprise_upload")
        # Solo restauramos el resultado tras validar de nuevo la misma entrada.
        # Así incluso una interrupción al preparar la imagen invalida el anterior.
        prediction = st.session_state.pop("enterprise_prediction", None)
        picture = None
        prediction_key = None
        if upload:
            try:
                picture = web_image(upload.getvalue())
                prediction_key = _classification_key(picture, client, images, documents)
            except (OSError, ValueError, GPUClientError, Image.DecompressionBombError) as exc:
                st.error(str(exc))
        if prediction and prediction_key is not None and prediction[0] == prediction_key:
            st.session_state["enterprise_prediction"] = prediction
        if (upload and st.button("Clasificar imagen nueva", type="primary", key="enterprise_predict")
                and picture is not None):
            # Un reintento fallido no debe dejar visible el éxito anterior.
            st.session_state.pop("enterprise_prediction", None)
            try:
                vector = unit_matrix(client.embed([{"image_base64": encode_image(picture)}]))[0]
                st.session_state["enterprise_prediction"] = (
                    prediction_key, picture, classify(vector, images, documents),
                )
            except (OSError, ValueError, GPUClientError) as exc:
                st.error(str(exc))
        prediction = st.session_state.get("enterprise_prediction")
        if prediction and prediction_key is not None and prediction[0] == prediction_key:
            st.image(prediction[1], width=280)
            best_label, best_score = prediction[2][0]
            in_sample_floor = min(row["cosine"] for row in outcomes)
            if best_score < in_sample_floor:
                st.warning(f"Imagen posiblemente fuera de esta colección: su mejor coseno es "
                           f"{best_score:.3f}; el menor entre las páginas evaluadas fue "
                           f"{in_sample_floor:.3f}. No asignaría un tipo automáticamente.")
            else:
                st.success(f"Centro más cercano: {display_type(best_label)} "
                           f"(coseno {best_score:.3f})")
            st.dataframe([{"Tipo": display_type(label), "Coseno": round(score, 3)}
                          for label, score in prediction[2][:5]], hide_index=True, width="stretch")
            st.caption("El umbral usa el mínimo de esta misma muestra; es solo una señal exploratoria. "
                       "Los cosenos no son probabilidades ni confianza calibrada.")
        st.markdown("#### Busca documentos con una frase")
        query = st.text_input("Consulta", "recibo de compra con total e impuestos", key="enterprise_query")
        if st.button("Buscar con texto", key="enterprise_search"):
            try:
                query_vector = unit_matrix(client.embed(
                    [{"text": query}], instruction="Retrieve images relevant to the user's query."
                ))[0]
                scores = images @ query_vector
                st.session_state["enterprise_search_result"] = (query, np.argsort(scores)[::-1][:5], scores)
            except (ValueError, GPUClientError) as exc:
                st.error(str(exc))
        result = st.session_state.get("enterprise_search_result")
        if result and result[0] == query:
            for position, index in enumerate(result[1], start=1):
                found = documents[int(index)]
                left, right = st.columns([1, 4], vertical_alignment="center")
                with left:
                    st.image(image_bytes(found), width=110)
                with right:
                    st.write(f"**{position}. {found['id']}** · {display_type(found['label'])}")
                    st.caption(f"Coseno texto ↔ imagen: {result[2][index]:.3f}")
    with inspect_tab:
        st.subheader("Seguir señales por 28 capas")
        st.write("Qwen codifica la imagen y la transcripción por separado. "
                 "Aquí seguimos cómo cambia el token de resumen en las 28 capas del decodificador de lenguaje.")
        if st.button("Inspeccionar imagen y transcripción", type="primary", key="enterprise_inspect"):
            try:
                picture = web_image(image_bytes(document), longest=512)
                with st.spinner("Kratos registra las capas de ambas modalidades…"):
                    scans = {
                        "imagen": client.inspect({"image_base64": encode_image(picture)}),
                        "transcripción": client.inspect({"text": document["text"][:1000]}),
                    }
                st.session_state["enterprise_scans"] = (document["id"], scans)
            except (OSError, ValueError, GPUClientError) as exc:
                st.error(str(exc))
        scans_result = st.session_state.get("enterprise_scans")
        if scans_result and scans_result[0] == document["id"]:
            st.plotly_chart(layer_chart(scans_result[1]), width="stretch")
            st.caption("1 − coseno compara el token de resumen entre capas contiguas. "
                       "No mide causalidad ni muestra todas las activaciones del codificador visual.")
    st.caption("Fuente de la muestra y condiciones por fila en el manifiesto local `data/empresa/manifest.json`.")
