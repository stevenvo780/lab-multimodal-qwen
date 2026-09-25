"""Embeddings multimodales con Qwen3-VL-Embedding (texto e imagen en el mismo espacio).

    from embed import *
    t = embed_text(["un perro en la playa", "una ciudad de noche"])
    i = embed_image(["samples/perro.jpg"])
    cos(t, i)          # similitud coseno (los vectores ya salen normalizados)

Modelo por defecto: models/Qwen3-VL-Embedding-2B. Para el 8B: TK_MODEL=Qwen/Qwen3-VL-Embedding-8B
"""
import os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("TK_MODEL", os.path.join(HERE, "models", "Qwen3-VL-Embedding-2B"))
_local = MODEL if os.path.isdir(MODEL) else os.path.join(HERE, "models", "Qwen3-VL-Embedding-2B")
sys.path.insert(0, _local)  # trae scripts/qwen3_vl_embedding.py del repo del modelo
from scripts.qwen3_vl_embedding import Qwen3VLEmbedder  # noqa: E402

_modelo = None


def modelo():
    global _modelo
    if _modelo is None:
        _modelo = Qwen3VLEmbedder(model_name_or_path=MODEL, torch_dtype=torch.bfloat16)
    return _modelo


def _norm(e):
    e = torch.as_tensor(e).float()
    return torch.nn.functional.normalize(e, dim=-1).cpu().numpy()


def _img(p):
    p = str(p)
    return p if p.startswith(("http://", "https://", "file://")) else os.path.abspath(p)


def embed(items, instruction=None, lote=8):
    """items: lista de dicts {"text": ...} / {"image": ...} / ambos."""
    out = []
    for k in range(0, len(items), lote):
        chunk = items[k:k + lote]
        if instruction:
            chunk = [dict(it, instruction=instruction) for it in chunk]
        with torch.inference_mode():
            out.append(_norm(modelo().process(chunk)))
    return np.concatenate(out) if out else np.zeros((0, 0))


def embed_text(texts, instruction=None):
    return embed([{"text": t} for t in texts], instruction)


def embed_image(images, instruction=None):
    return embed([{"image": _img(p)} for p in images], instruction)


def inspect(item, instruction=None):
    """Resume el último token tras cada capa del decoder durante una inferencia.

    Los hooks copian solo un vector de 2048 valores a CPU por capa. No se
    conservan activaciones completas ni se vuelve a cargar el modelo.
    """
    loaded = modelo()
    conversation = loaded.format_model_input(
        text=item.get("text"), image=item.get("image"), instruction=instruction
    )
    inputs = loaded._preprocess_inputs([conversation])
    token_count = int(inputs["attention_mask"].sum().item())
    inputs = {key: value.to(loaded.model.device) for key, value in inputs.items()}
    layers = []
    previous = None
    handles = []

    def record(layer_number):
        def hook(_module, _arguments, output):
            nonlocal previous
            hidden = output[0] if isinstance(output, tuple) else output
            # Solo el último token; copy=True evita mantener el tensor de la capa.
            vector = hidden[0, token_count - 1].detach().to(
                device="cpu", dtype=torch.float32, copy=True
            )
            delta_cos = None
            if previous is not None:
                delta_cos = float(torch.nn.functional.cosine_similarity(
                    vector, previous, dim=0
                ).item())
            layers.append({
                "layer": layer_number,
                "rms": float(vector.square().mean().sqrt().item()),
                "mean": float(vector.mean().item()),
                "std": float(vector.std(unbiased=False).item()),
                "delta_cos": delta_cos,
            })
            previous = vector
        return hook

    try:
        for number, layer in enumerate(loaded.model.language_model.layers, start=1):
            handles.append(layer.register_forward_hook(record(number)))
        with torch.inference_mode():
            output = loaded.forward(inputs)
            pooled = loaded._pooling_last(
                output["last_hidden_state"], output["attention_mask"]
            )
            dimension = int(pooled.shape[-1])
    finally:
        for handle in handles:
            handle.remove()

    return {
        "model": os.path.basename(MODEL.rstrip("/")),
        "dimension": dimension,
        "modality": "image" if "image" in item else "text",
        "input_token_count": token_count,
        "layers": layers,
    }


def cos(a, b):
    return np.asarray(a) @ np.asarray(b).T


def cronometro(f, *a, **k):
    t = time.time(); r = f(*a, **k)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return r, time.time() - t


__all__ = ["embed", "embed_text", "embed_image", "inspect", "cos", "modelo", "cronometro", "np", "torch"]
