# Laboratorio visual de embeddings multimodales

Proyecto local para seguir una charla sobre Qwen3-VL, Python y Streamlit. El **atlas y laboratorio integrado** corren en este equipo. Solo la inferencia del modelo corre en una GPU remota, a través de un túnel SSH.

## Abrir con un comando

```bash
cd /ruta/a/lab-multimodal-qwen
./abrir-lab.sh
```

Abre solo el [atlas SPA](http://127.0.0.1:4173). Su navegación integra el comparador multimodal, el caso de documentos, la aritmética vectorial y la tomografía en la misma página. Streamlit corre localmente en el puerto 8501 como motor de los módulos; el navegador no tiene que saltar entre pestañas. El script comprueba o inicia el procesador GPU remoto, abre el túnel `127.0.0.1:18765 → GPU remota:127.0.0.1:8765` y sirve la SPA localmente. Para detener la sesión local, pulsa Ctrl+C.

La primera vez, prepara el entorno Python local con `uv sync --python 3.12`. El proyecto usa `pyproject.toml` y `uv.lock`; **no instala PyTorch ni descarga el modelo en el portátil**. El valor predeterminado del SSH es `kratos`; puedes ajustarlo con `LAB_SSH_HOST=mi-servidor LAB_REMOTE_DIR=/ruta/remota ./abrir-lab.sh`. Si cambias el puerto de la SPA, ejecuta `LAB_WEB_PORT=4174 ./abrir-lab.sh`.

## Dónde está el código

| Componente | Ruta | Se ejecuta en |
| --- | --- | --- |
| SPA, catálogo y bitácora | `index.html`, `styles.css`, `content.js`, `app.js` | Local, puerto 4173 |
| Laboratorio Python | `laboratorio.py`, `comparador_lab.py`, `caso_empresa.py`, `tomografia_lab.py`, `gpu_client.py` | Local, puerto 8501 |
| Muestra de documentos y embeddings | `data/empresa/` | Local; imágenes y vectores derivados con GPU remota |
| Fotos didácticas reproducibles | `samples/`, `preparar_samples.py`, `ASSETS.md` | Local; imágenes CC0 verificadas en Commons |
| Embeddings e inspección de capas | `gpu_remote/gpu_server.py`, `gpu_remote/embed.py` | GPU de kratos, puerto 8765 |
| Copia para VS Code en la torre | Valor de `LAB_REMOTE_DIR` | GPU remota |

La copia remota existe para inspeccionar y ejecutar el código en VS Code. El código del procesador también queda en `gpu_remote/` dentro de este proyecto local. En la instalación original, el modelo y el entorno pesado se reutilizan desde otro laboratorio de la torre mediante enlaces, sin duplicar varios gigabytes. Otras instalaciones deben preparar su propio entorno GPU y el modelo.

Las 17 fotos del laboratorio se distribuyen con [procedencia y crédito por archivo](ASSETS.md). `preparar_samples.py` permite regenerarlas y verifica que las fichas originales sigan declarando CC0 antes de descargarlas. Las notas personales y el dataset documental derivado permanecen solo en el equipo local y están excluidos de Git.

Si editas el procesador aquí, compara y sincroniza los archivos con `./sincronizar-gpu.sh check` y `./sincronizar-gpu.sh push`. Si editas en VS Code Remote-SSH, usa `./sincronizar-gpu.sh pull` para traer esos cambios al proyecto local. La sincronización conserva una copia de seguridad de archivos reemplazados y **no reinicia automáticamente** el proceso GPU; después de editar Python, entra a `embedding-gpu`, detén el servidor con Ctrl+C y ejecútalo otra vez con `CUDA_VISIBLE_DEVICES=0 .venv/bin/python gpu_server.py`.

Para mirar la sesión del procesador en la torre configurada:

```bash
ssh -t "${LAB_SSH_HOST:-kratos}" 'tmux attach -t embedding-gpu'
```

## Ruta de aprendizaje

1. **Catálogo:** identifica modalidad, encoder, embedding, normalización, similitud coseno, ranking, PCA y UMAP.
2. **Matriz texto↔imagen:** escribe descripciones y compara filas y columnas. La puntuación coseno expresa cercanía relativa, no una probabilidad.
3. **Búsqueda:** cambia la consulta y observa cómo cambia el `top-k`.
4. **Aritmética vectorial:** prueba `A − B + C`; registra cuándo la analogía funciona o falla.
5. **Proyección 2D:** compara PCA y UMAP. Una posición en 2D resume y puede deformar relaciones del espacio original.
6. **Bitácora:** anota hipótesis, resultado e interpretación en el catálogo, exporta tus notas y compártelas en la conversación para replicar la idea en Python.

## Comparador multimodal con tus datos

En el comparador asigna **texto, imagen o ambos** a A, B y C. Puedes subir hasta 20 imágenes adicionales y escribir hasta 30 textos. El triángulo 3D de A/B/C conserva exactamente los ángulos y distancias entre esos tres vectores normalizados de 2048 dimensiones. El mapa PCA 3D muestra la colección completa, pero sus distancias visibles son aproximadas. La tabla y los vecinos usan el coseno original en 2048D. Los embeddings nuevos se calculan en kratos; la proyección y la búsqueda corren localmente.

## Caso de documentos de empresa

El caso contiene **51 páginas y 10 tipos** de una [colección pública de OCR](https://huggingface.co/datasets/ilsilfverskiold/ocr-benchmark): [OmniAI](https://huggingface.co/datasets/getomni-ai/ocr-benchmark), [SROIE](https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE) y [SynFinTabs](https://huggingface.co/datasets/ethanbradley/synfintabs). Mezcla páginas escaneadas reales y sintéticas. El texto de cada fila es una **transcripción de referencia**, no OCR producido por Qwen. La licencia declarada de cada fuente y la procedencia de cada documento están en `data/empresa/manifest.json`. Es una muestra local didáctica; verifica derechos de imágenes individuales antes de reutilizarlas en otro contexto.

El árbol expande vecinos de un documento usando el coseno de embeddings de imagen en 2048D. El mapa PCA pone imágenes y transcripciones en el mismo espacio 3D y permite girar los puntos. La clasificación compara con centros por tipo y excluye la página evaluada de su propio centro. Como hay solo cinco o seis páginas por tipo y tipos asociados a fuentes/plantillas, sus aciertos no predicen rendimiento con documentos nuevos. Puedes subir una imagen para clasificarla, buscar páginas con texto e inspeccionar las 28 capas para un par imagen/transcripción.

Para reconstruir la muestra y sus vectores desde la colección de origen:

```bash
.venv/bin/python preparar_caso_empresa.py --extraer --embeber
```

El parquet original ocupa unos 94 MiB; `data/` está ignorado por Git. La extracción conserva 51 JPEG reducidos y un manifiesto con identificador, tipo, fuente, licencia, división, índice, tamaño de dificultad y SHA-256. El cálculo remoto guarda 102 embeddings (imagen y transcripción) de forma reanudable en `data/empresa/embeddings.json`.

## Tres vectores en 3D y tomografía

La vista A, B y C usa **embeddings reales** de las entradas que elijas. Calcula sus tres cosenos y ángulos en 2048 dimensiones y los coloca en 3D a partir de su matriz de Gram; esa representación conserva exactamente los tres ángulos. También muestra `R = normalizar(A − B + C)` y sus vecinos. Los ejes 3D se eligen para dibujar, no corresponden a propiedades semánticas independientes.

La tomografía tiene dos escalas. Primero muestra el mapa PCA de 22 embeddings finales de texto e imagen. Luego inspecciona la trayectoria del token de resumen en **28 capas del decodificador** para una frase y una imagen: magnitud RMS y coseno entre capas consecutivas. Una imagen pasa antes por el codificador visual; estas curvas describen activaciones internas, no atención ni una explicación causal. El servidor remoto envía solo estadísticas resumidas.

La vista de entrenamiento ajusta una **sonda lineal local** con 16 embeddings congelados; deja seis ejemplos fuera para comprobar predicciones. Qwen no cambia sus pesos. La curva de pérdida y las puntuaciones de la sonda sirven para estudiar el mecanismo, no para medir de forma fiable el rendimiento general del modelo.

## Repetir directamente en Python remoto

En la torre, desde el directorio indicado por `LAB_REMOTE_DIR` y con su entorno Python:

```python
from embed import embed_text, embed_image, cos

consulta = embed_text(["un perro en la playa"])
imagenes = embed_image(["samples/perro.jpg", "samples/gato.jpg"])
print(consulta.shape, imagenes.shape)
print(cos(consulta, imagenes))
```

`embed.py` devuelve vectores normalizados; por eso el producto punto equivale a similitud coseno. **Qwen3-VL-Embedding-2B crea vectores; no genera respuestas de chat.** Una futura fase de charla con respuestas generadas requeriría otro modelo, como Qwen3-VL-Instruct.

## Pruebas del cliente sin GPU

Con el entorno local de Python 3.12 ya preparado y `requests` disponible:

```bash
uv run --offline --no-sync python -m unittest discover -s tests -v
```

Las pruebas usan `unittest` de Python y respuestas HTTP simuladas; no necesitan
dependencias de prueba adicionales, GPU, túnel SSH, servicios, pesos ni datasets.
Sustituyen el transporte HTTP y bloquean conexiones y consultas DNS durante cada
prueba. Comprueban `/health`, lotes y orden de embeddings, limpieza de entradas e
instrucciones, errores HTTP/JSON/timeout, dimensiones, valores no finitos y las
28 capas de `/inspect`. Esto valida el contrato del cliente, no la inferencia real
ni una integración de extremo a extremo con la GPU.

## Fuentes técnicas

- [Repositorio oficial de Qwen3-VL-Embedding](https://github.com/QwenLM/Qwen3-VL-Embedding)
- [Ficha oficial del modelo Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B)
- [Ficha oficial de Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
- [Documentación de Streamlit](https://docs.streamlit.io/)
