"""Descarga muestras didácticas CC0 verificadas desde Wikimedia Commons.

Uso: .venv/bin/python preparar_samples.py
"""

from __future__ import annotations

import hashlib
import html
import io
import re
import time
from pathlib import Path

import requests
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parent
API = "https://commons.wikimedia.org/w/api.php"
LICENSE = "https://creativecommons.org/publicdomain/zero/1.0/"
SAMPLES = {
    "perro.jpg": "Dog in Chacachacare.jpg",
    "gato.jpg": "Tabby cat with blue eyes-3336579.jpg",
    "leon.jpg": "Lion Ngorongoro 2.jpg",
    "panda.jpg": "Giant panda in Edinburgh Zoo 01.jpg",
    "pizza.jpg": "Tomato onion pizza.jpg",
    "banano.jpg": "Banana for human nutrition.jpg",
    "cafe.jpg": "Cup of coffee in Café Butter - Prenzlauer Berg, Berlin.jpg",
    "manzana.jpg": "Apples fruit.jpg",
    "coliseo.jpg": "Colosseum of Rome, Italy.jpg",
    "torre-eiffel.jpg": "Tour Eiffel vue du Champ-de-Mars.jpg",
    "everest.jpg": "GoWildImages MtEverest NEP0555.jpg",
    "bicicleta.jpg": "Bicycle 02.jpg",
    "avion.jpg": "September 2006, airplane in Reykjavik.jpg",
    "guitarra.jpg": "Guitar of the week 28.jpg",
    "playa-perro.jpg": "Dog on a beach.jpg",
    "girasol.jpg": "Sunflowers public domain helianthus annuus IMG 2465.jpg",
    "tigre.jpg": "Tijger (Panthera tigris).jpg",
}


def get(session: requests.Session, url: str, **kwargs) -> requests.Response:
    for attempt in range(7):
        response = session.get(url, timeout=45, **kwargs)
        if response.status_code not in (429, 500, 502, 503, 504):
            response.raise_for_status()
            return response
        wait = min(30, 2 ** attempt)
        print(f"HTTP {response.status_code}: reintento en {wait} s…")
        time.sleep(wait)
    response.raise_for_status()
    return response


def plain_text(value: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]*>", " ", value)).split())


def metadata(session: requests.Session) -> dict[str, dict]:
    titles = "|".join(f"File:{title}" for title in SAMPLES.values())
    payload = get(
        session,
        API,
        params={
            "action": "query",
            "titles": titles,
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|size|sha1|mime",
            "iiurlwidth": "960",
            "format": "json",
        },
    ).json()
    pages = {page["title"]: page for page in payload["query"]["pages"].values()}
    result = {}
    for filename, title in SAMPLES.items():
        page = pages.get(f"File:{title}")
        if not page or "missing" in page or not page.get("imageinfo"):
            raise RuntimeError(f"No encuentro la fuente en Commons: {title}")
        info = page["imageinfo"][0]
        ext = info.get("extmetadata", {})
        license_name = ext.get("LicenseShortName", {}).get("value", "")
        if license_name != "CC0":
            raise RuntimeError(f"Licencia inesperada para {title}: {license_name!r}")
        if not info.get("mime", "").startswith("image/"):
            raise RuntimeError(f"Fuente no compatible con imagen: {title}")
        result[filename] = {
            "title": title,
            "page": info["descriptionurl"],
            "download": info.get("thumburl", info["url"]),
            "sha1": info.get("sha1", ""),
            "author": plain_text(ext.get("Artist", {}).get("value", "Autor no indicado")),
            "license": license_name,
        }
    return result


def save_image(session: requests.Session, url: str, destination: Path) -> None:
    payload = get(session, url).content
    with Image.open(io.BytesIO(payload)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((960, 960), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=82, optimize=True, progressive=True)
    destination.write_bytes(output.getvalue())


def write_attribution(info: dict[str, dict]) -> None:
    lines = [
        "# Fuentes de las imágenes de ejemplo",
        "",
        "Las 17 fotografías de `samples/` proceden de Wikimedia Commons. El script",
        "[`preparar_samples.py`](preparar_samples.py) comprueba por API que cada archivo",
        "sigue publicado como [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)",
        "antes de descargarlo. Las copias locales están redimensionadas a un máximo",
        "de 960 píxeles y convertidas a JPEG para el laboratorio. Se acredita a los",
        "autores aunque CC0 no exija atribución.",
        "",
        "| Archivo local | Fuente original | Autor indicado en Commons | Licencia |",
        "| --- | --- | --- | --- |",
    ]
    for filename, item in info.items():
        title = item["title"].replace("|", r"\|")
        author = item["author"].replace("|", r"\|")
        lines.append(
            f"| `samples/{filename}` | [{title}]({item['page']}) | "
            f"{author} | [CC0 1.0]({LICENSE}) |"
        )
    lines += [
        "",
        "Para regenerar las muestras: `.venv/bin/python preparar_samples.py`.",
        "La licencia y la atribución corresponden a cada página original, no al",
        "nombre del archivo local. Si Commons cambia la licencia o retira alguna",
        "imagen, el script se detiene antes de sustituir las copias locales.",
        "",
    ]
    (ROOT / "ASSETS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    session = requests.Session()
    session.headers["User-Agent"] = (
        "MultimodalLearningLab/1.0 (educational sample downloader; Wikimedia Commons API)"
    )
    info = metadata(session)
    sample_dir = ROOT / "samples"
    sample_dir.mkdir(exist_ok=True)
    for filename, item in info.items():
        path = sample_dir / filename
        save_image(session, item["download"], path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        print(f"{filename:22} {path.stat().st_size // 1024:4} KiB  SHA-256 {digest}")
    write_attribution(info)
    print("ASSETS.md actualizado; todas las fuentes verificadas como CC0.")


if __name__ == "__main__":
    main()
