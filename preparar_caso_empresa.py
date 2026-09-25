"""Extrae páginas de fuentes identificadas y calcula embeddings con kratos.

Uso: .venv/bin/python preparar_caso_empresa.py [--extraer] [--embeber]
Las 51 páginas permitidas son una muestra didáctica, no un benchmark empresarial.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path

from PIL import Image, ImageOps

from gpu_client import GPUClient, encode_image


ROOT = Path(__file__).resolve().parent
PARQUET = ROOT / "data" / "ocr-benchmark" / "train.parquet"
CASE = ROOT / "data" / "empresa"
MANIFEST = CASE / "manifest.json"
EMBEDDINGS = CASE / "embeddings.json"
SOURCE_URL = (
    "https://huggingface.co/datasets/ilsilfverskiold/ocr-benchmark/resolve/main/"
    "data/train-00000-of-00001.parquet"
)
SOURCES = {
    "getomni-ai/ocr-benchmark": {
        "license": "MIT",
        "url": "https://huggingface.co/datasets/getomni-ai/ocr-benchmark",
    },
    "jsdnrs/ICDAR2019-SROIE": {
        "license": "CC-BY-4.0",
        "url": "https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE",
    },
    "ethanbradley/synfintabs": {
        "license": "MIT",
        "url": "https://huggingface.co/datasets/ethanbradley/synfintabs",
    },
}


def download_parquet() -> None:
    if PARQUET.exists():
        return
    import requests

    PARQUET.parent.mkdir(parents=True, exist_ok=True)
    temporary = PARQUET.with_suffix(".part")
    print(f"Descargando {SOURCE_URL}", flush=True)
    with requests.get(SOURCE_URL, stream=True, timeout=(15, 180)) as response:
        response.raise_for_status()
        with temporary.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)
    temporary.replace(PARQUET)


def prepare() -> list[dict]:
    import pyarrow.parquet as pq

    download_parquet()
    CASE.mkdir(parents=True, exist_ok=True)
    images = CASE / "images"
    images.mkdir(exist_ok=True)
    documents: list[dict] = []
    for row in pq.read_table(PARQUET).to_pylist():
        source_meta = json.loads(row["structured_gt"] or "{}")
        source = source_meta.get("dataset")
        if source not in SOURCES:
            continue
        image_bytes = row["image"]["bytes"]
        if not image_bytes:
            raise ValueError(f"Sin imagen: {row['stem']}")
        with Image.open(io.BytesIO(image_bytes)) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
            image.thumbnail((768, 768), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, "JPEG", quality=82, optimize=True)
        picture = buffer.getvalue()
        ident = row["stem"]
        path = images / f"{ident}.jpg"
        path.write_bytes(picture)
        documents.append({
            "id": ident,
            "label": row["doc_type"],
            "tier": row["tier"],
            "source": source,
            "source_split": source_meta.get("split"),
            "source_index": source_meta.get("dataset_offset"),
            "source_url": SOURCES[source]["url"],
            "license": SOURCES[source]["license"],
            "text": row["gt_text"] or "",
            "image": f"images/{path.name}",
            "sha256": hashlib.sha256(picture).hexdigest(),
        })
    documents.sort(key=lambda item: (item["label"], item["id"]))
    if len(documents) != 51:
        raise ValueError(f"Se esperaban 51 páginas con procedencia clara; encontradas {len(documents)}")
    MANIFEST.write_text(json.dumps({
        "description": "Muestra didáctica de documentos públicos reales y sintéticos",
        "collection": "ilsilfverskiold/ocr-benchmark",
        "collection_url": "https://huggingface.co/datasets/ilsilfverskiold/ocr-benchmark",
        "documents": documents,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Preparados {len(documents)} documentos en {CASE}", flush=True)
    return documents


def save_vectors(vectors: dict) -> None:
    temporary = EMBEDDINGS.with_suffix(".tmp")
    temporary.write_text(json.dumps(vectors, separators=(",", ":")), encoding="utf-8")
    os.replace(temporary, EMBEDDINGS)


def embed(documents: list[dict]) -> None:
    client = GPUClient()
    client.health()
    vectors = json.loads(EMBEDDINGS.read_text(encoding="utf-8")) if EMBEDDINGS.exists() else {}
    for modality in ("image", "text"):
        todo = [document for document in documents if modality not in vectors.get(document["id"], {})]
        print(f"{modality}: {len(todo)} pendientes", flush=True)
        for start in range(0, len(todo), client.batch_size):
            batch = todo[start:start + client.batch_size]
            if modality == "image":
                items = [{"image_base64": encode_image((CASE / doc["image"]).read_bytes())}
                         for doc in batch]
            else:
                items = [{"text": doc["text"][:1200] or doc["label"]} for doc in batch]
            result = client.embed(items)
            for doc, vector in zip(batch, result, strict=True):
                vectors.setdefault(doc["id"], {})[modality] = vector
            save_vectors(vectors)
            print(f"{modality}: {min(start + len(batch), len(todo))}/{len(todo)}", flush=True)
    print(f"Embeddings completos: {len(vectors)} documentos", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extraer", action="store_true", help="Descargar y extraer imágenes")
    parser.add_argument("--embeber", action="store_true", help="Calcular embeddings en kratos")
    options = parser.parse_args()
    if options.extraer or not MANIFEST.exists():
        documents = prepare()
    else:
        documents = json.loads(MANIFEST.read_text(encoding="utf-8"))["documents"]
    if options.embeber:
        embed(documents)


if __name__ == "__main__":
    main()
