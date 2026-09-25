"""Cliente pequeño para la API de embeddings de kratos por túnel SSH.

La web local no carga pesos. Envía texto y/o imágenes codificadas en base64 a
127.0.0.1:18765, que `abrir-lab.sh` comunica con el servicio GPU remoto.
"""

from __future__ import annotations

import base64
import math
import os
from collections.abc import Mapping, Sequence
from typing import Any

import requests


DEFAULT_API_URL = "http://127.0.0.1:18765"
EXPECTED_DIMENSION = 2048


class GPUClientError(RuntimeError):
    """Error de conexión, inferencia o respuesta inválida de la API GPU."""


def encode_image(image_bytes: bytes) -> str:
    """Codifica los bytes de una imagen para el campo `image_base64`."""
    if not image_bytes:
        raise GPUClientError("La imagen está vacía.")
    return base64.b64encode(image_bytes).decode("ascii")


class GPUClient:
    def __init__(
        self,
        base_url: str | None = None,
        *,
        health_timeout: float = 3.0,
        embed_timeout: float = 180.0,
        batch_size: int = 4,
    ) -> None:
        self.base_url = (base_url or os.getenv("QWEN_GPU_API_URL") or DEFAULT_API_URL).rstrip("/")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("QWEN_GPU_API_URL debe comenzar con http:// o https://")
        if batch_size < 1:
            raise ValueError("batch_size debe ser positivo")
        self.health_timeout = health_timeout
        self.embed_timeout = embed_timeout
        self.batch_size = batch_size

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = requests.request(method, f"{self.base_url}{path}", **kwargs)
            response.raise_for_status()
            data = response.json()
        except requests.Timeout as exc:
            raise GPUClientError("La GPU tardó demasiado en responder. Prueba con menos imágenes.") from exc
        except requests.ConnectionError as exc:
            raise GPUClientError(
                "No hay conexión con la GPU. Abre el túnel SSH a kratos y comprueba el servicio remoto."
            ) from exc
        except requests.HTTPError as exc:
            detail = ""
            try:
                payload = response.json()
                if isinstance(payload, dict):
                    detail = str(payload.get("detail", ""))
            except ValueError:
                detail = response.text[:250]
            status = response.status_code
            raise GPUClientError(f"La API GPU devolvió HTTP {status}. {detail}".strip()) from exc
        except (requests.RequestException, ValueError) as exc:
            raise GPUClientError(f"No se pudo leer la respuesta de la API GPU: {exc}") from exc
        if not isinstance(data, dict):
            raise GPUClientError("La API GPU devolvió un JSON inesperado.")
        return data

    def health(self) -> dict[str, Any]:
        data = self._request("GET", "/health", timeout=self.health_timeout)
        if data.get("status") != "ok":
            raise GPUClientError("El servicio GPU no indicó estado 'ok'.")
        if not isinstance(data.get("model_loaded"), bool):
            raise GPUClientError("La respuesta /health no contiene 'model_loaded' válido.")
        if data.get("dimension") != EXPECTED_DIMENSION:
            raise GPUClientError(
                f"Dimensión inesperada en /health: {data.get('dimension')!r}; "
                f"se esperaba {EXPECTED_DIMENSION}."
            )
        return data

    def embed(
        self,
        items: Sequence[Mapping[str, str]],
        *,
        instruction: str | None = None,
    ) -> list[list[float]]:
        """Envía texto, imagen o ambos y devuelve vectores en el mismo orden."""
        if not items:
            raise GPUClientError("Añade al menos una entrada para calcular embeddings.")
        clean_items: list[dict[str, str]] = []
        for index, item in enumerate(items, start=1):
            if not isinstance(item, Mapping):
                raise GPUClientError(f"La entrada {index} no es un objeto válido.")
            text = item.get("text")
            image = item.get("image_base64")
            if text is not None and not isinstance(text, str):
                raise GPUClientError(f"El texto de la entrada {index} debe ser una cadena.")
            if image is not None and not isinstance(image, str):
                raise GPUClientError(f"La imagen de la entrada {index} debe estar en base64.")
            clean: dict[str, str] = {}
            if text and text.strip():
                clean["text"] = text.strip()
            if image:
                clean["image_base64"] = image
            if not clean:
                raise GPUClientError(f"La entrada {index} necesita texto o imagen.")
            clean_items.append(clean)

        if instruction is not None and not isinstance(instruction, str):
            raise GPUClientError("La instrucción debe ser texto.")
        all_vectors: list[list[float]] = []
        for start in range(0, len(clean_items), self.batch_size):
            batch = clean_items[start : start + self.batch_size]
            payload: dict[str, Any] = {"items": batch}
            if instruction and instruction.strip():
                payload["instruction"] = instruction.strip()
            data = self._request(
                "POST",
                "/embed",
                json=payload,
                timeout=self.embed_timeout,
            )
            if data.get("dimension") != EXPECTED_DIMENSION:
                raise GPUClientError(
                    f"La API devolvió dimensión {data.get('dimension')!r}; "
                    f"se esperaba {EXPECTED_DIMENSION}."
                )
            vectors = data.get("vectors")
            if not isinstance(vectors, list) or len(vectors) != len(batch):
                raise GPUClientError("La API devolvió un número inesperado de vectores.")
            for position, vector in enumerate(vectors, start=start + 1):
                if not isinstance(vector, list) or len(vector) != EXPECTED_DIMENSION:
                    raise GPUClientError(f"El vector {position} tiene una dimensión inválida.")
                try:
                    numeric = [float(value) for value in vector]
                except (TypeError, ValueError) as exc:
                    raise GPUClientError(f"El vector {position} contiene datos no numéricos.") from exc
                if not all(math.isfinite(value) for value in numeric):
                    raise GPUClientError(f"El vector {position} contiene valores no finitos.")
                all_vectors.append(numeric)
        return all_vectors

    def inspect(self, item: Mapping[str, str]) -> dict[str, Any]:
        """Resume el token de salida en las 28 capas, sin transferir activaciones."""
        if not isinstance(item, Mapping):
            raise GPUClientError("La entrada de inspección debe ser texto o imagen.")
        text = item.get("text")
        image = item.get("image_base64")
        if bool(text) == bool(image):
            raise GPUClientError("Inspecciona una frase o una imagen por vez.")
        clean = {"text": text} if text else {"image_base64": image}
        data = self._request("POST", "/inspect", json={"item": clean}, timeout=self.embed_timeout)
        if data.get("dimension") != EXPECTED_DIMENSION:
            raise GPUClientError("La inspección devolvió una dimensión inesperada.")
        layers = data.get("layers")
        if not isinstance(layers, list) or len(layers) != 28:
            raise GPUClientError("La inspección no devolvió las 28 capas esperadas.")
        for expected, entry in enumerate(layers, start=1):
            if not isinstance(entry, dict) or entry.get("layer") != expected:
                raise GPUClientError("La inspección devolvió capas desordenadas.")
            for field in ("rms", "mean", "std"):
                value = entry.get(field)
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise GPUClientError(f"La capa {expected} no tiene {field} válido.")
            delta = entry.get("delta_cos")
            if expected > 1 and (not isinstance(delta, (int, float)) or not math.isfinite(delta)):
                raise GPUClientError(f"La capa {expected} no tiene coseno consecutivo válido.")
        return data
