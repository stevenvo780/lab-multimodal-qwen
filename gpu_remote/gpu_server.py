"""Servidor HTTP local de inferencia Qwen3-VL-Embedding para la interfaz en el portátil.

Solo escucha en 127.0.0.1 de Kratos. Se accede mediante un túnel SSH.
"""
from __future__ import annotations

import base64
import binascii
import io
import json
import logging
import tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from PIL import Image, UnidentifiedImageError

import embed

HOST = "127.0.0.1"
PORT = 8765
DIMENSION = 2048
MAX_ITEMS = 16
MAX_BODY_BYTES = 25 * 1024 * 1024
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
MAX_TEXT_CHARS = 10_000
MAX_INSTRUCTION_CHARS = 2_000
MAX_INSPECT_TEXT_CHARS = 1_000
MAX_INSPECT_IMAGE_BYTES = 1 * 1024 * 1024
MAX_INSPECT_IMAGE_SIDE = 512
IMAGE_EXTENSIONS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}


class APIError(Exception):
    def __init__(self, status: int, error: str, detail: str):
        super().__init__(detail)
        self.status = status
        self.error = error
        self.detail = detail


def decode_image(encoded: str, directory: Path, index: int, inspect: bool = False) -> str:
    if not isinstance(encoded, str) or not encoded:
        raise APIError(400, "invalid_image", f"items[{index}].image_base64 debe ser un texto base64.")
    if encoded.startswith("data:"):
        if ";base64," not in encoded:
            raise APIError(400, "invalid_image", f"items[{index}] tiene un data URL sin base64.")
        encoded = encoded.split(",", 1)[1]
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise APIError(400, "invalid_image", f"items[{index}] contiene base64 inválido.") from exc
    max_bytes = MAX_INSPECT_IMAGE_BYTES if inspect else MAX_IMAGE_BYTES
    if not data or len(data) > max_bytes:
        limit = "1 MiB" if inspect else "8 MiB"
        raise APIError(413, "image_too_large", f"items[{index}] debe contener una imagen de 1 byte a {limit}.")
    try:
        with Image.open(io.BytesIO(data)) as image:
            image_format = image.format
            width, height = image.size
            if image_format not in IMAGE_EXTENSIONS:
                raise APIError(400, "invalid_image", f"items[{index}] debe ser JPEG, PNG o WebP.")
            if inspect and max(width, height) > MAX_INSPECT_IMAGE_SIDE:
                raise APIError(413, "image_too_large", f"items[{index}] no puede superar 512 píxeles por lado en /inspect.")
            if width * height > MAX_IMAGE_PIXELS:
                raise APIError(413, "image_too_large", f"items[{index}] supera 20 megapíxeles.")
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise APIError(400, "invalid_image", f"items[{index}] no contiene una imagen válida.") from exc
    path = directory / f"image-{index}{IMAGE_EXTENSIONS[image_format]}"
    path.write_bytes(data)
    return str(path)


def prepare_items(payload: object, directory: Path, inspect: bool = False) -> tuple[list[dict[str, str]], str | None]:
    if not isinstance(payload, dict):
        raise APIError(400, "invalid_request", "El cuerpo debe ser un objeto JSON.")
    raw_items = payload.get("items")
    max_items = 1 if inspect else MAX_ITEMS
    if not isinstance(raw_items, list) or not 1 <= len(raw_items) <= max_items:
        raise APIError(400, "invalid_items", f"items debe ser una lista de 1 a {max_items} elementos.")
    instruction = payload.get("instruction")
    if instruction is not None:
        if not isinstance(instruction, str) or len(instruction) > MAX_INSTRUCTION_CHARS:
            raise APIError(400, "invalid_instruction", "instruction debe ser texto de hasta 2000 caracteres.")
        instruction = instruction or None
    items = []
    for index, raw in enumerate(raw_items):
        if not isinstance(raw, dict):
            raise APIError(400, "invalid_item", f"items[{index}] debe ser un objeto.")
        item = {}
        if "text" in raw:
            text = raw["text"]
            max_text_chars = MAX_INSPECT_TEXT_CHARS if inspect else MAX_TEXT_CHARS
            if not isinstance(text, str) or not text.strip() or len(text) > max_text_chars:
                raise APIError(400, "invalid_text", f"items[{index}].text debe contener de 1 a {max_text_chars} caracteres.")
            item["text"] = text
        if "image_base64" in raw:
            item["image"] = decode_image(raw["image_base64"], directory, index, inspect=inspect)
        if not item:
            raise APIError(400, "invalid_item", f"items[{index}] necesita text, image_base64 o ambos.")
        if inspect and len(item) != 1:
            raise APIError(400, "invalid_item", f"items[{index}] debe contener solo text o image_base64 en /inspect.")
        items.append(item)
    return items, instruction


class Handler(BaseHTTPRequestHandler):
    def send_json(self, status: int, data: dict) -> None:
        body = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.split("?", 1)[0] != "/health":
            self.send_json(404, {"error": "not_found", "detail": "Usa GET /health, POST /embed o POST /inspect."})
            return
        self.send_json(200, {
            "status": "ok",
            "model_loaded": embed._modelo is not None,
            "model": Path(embed.MODEL).name,
            "dimension": DIMENSION,
            "device": "cuda:0" if embed.torch.cuda.is_available() else "cpu",
        })

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path not in ("/embed", "/inspect"):
            self.send_json(404, {"error": "not_found", "detail": "Usa GET /health, POST /embed o POST /inspect."})
            return
        try:
            if not self.headers.get("Content-Type", "").lower().startswith("application/json"):
                raise APIError(415, "unsupported_media_type", "Content-Type debe ser application/json.")
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError as exc:
                raise APIError(411, "length_required", "Se requiere Content-Length.") from exc
            if length <= 0:
                raise APIError(400, "invalid_request", "El cuerpo JSON está vacío.")
            if length > MAX_BODY_BYTES:
                raise APIError(413, "request_too_large", "El cuerpo JSON supera 25 MiB.")
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise APIError(400, "invalid_json", "El cuerpo debe contener JSON UTF-8 válido.") from exc
            with tempfile.TemporaryDirectory(prefix="qwen3-vl-") as temp:
                if path == "/inspect":
                    if not isinstance(payload, dict) or "item" not in payload:
                        raise APIError(400, "invalid_item", "El cuerpo de /inspect necesita un objeto item.")
                    inspect_payload = {"items": [payload["item"]], "instruction": payload.get("instruction")}
                    items, instruction = prepare_items(inspect_payload, Path(temp), inspect=True)
                    result = embed.inspect(items[0], instruction=instruction)
                else:
                    items, instruction = prepare_items(payload, Path(temp))
                    vectors = embed.embed(items, instruction=instruction)
                    result = {
                        "vectors": vectors.tolist(),
                        "dimension": int(vectors.shape[1]),
                    }
            self.send_json(200, result)
        except APIError as exc:
            self.send_json(exc.status, {"error": exc.error, "detail": exc.detail})
        except Exception as exc:
            logging.exception("Fallo durante la inferencia")
            self.send_json(500, {"error": "inference_failed", "detail": str(exc)})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    server = HTTPServer((HOST, PORT), Handler)
    print(f"Qwen3-VL-Embedding API en http://{HOST}:{PORT}", flush=True)
    server.serve_forever()
