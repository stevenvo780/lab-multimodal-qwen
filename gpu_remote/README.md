# Laboratorio de embeddings multimodales · GPU en Kratos

Este directorio es el workspace **editable desde VS Code Remote-SSH**:
 /datos/workspaces/personal/conferencias/EmbedingMultimodales

La interfaz Streamlit se ejecuta en el equipo local. Aquí solo se calcula la inferencia
en la GPU con Qwen3-VL-Embedding-2B. La comunicación sigue esta ruta:

~~~text
Streamlit local → 127.0.0.1:18765 → túnel SSH → 127.0.0.1:8765 en Kratos
                                              → gpu_server.py → embed.py → modelo
~~~

## Ejecutar en Kratos

Conecta VS Code a kratos mediante Remote-SSH y abre este directorio. En una terminal
del workspace:

~~~bash
./start-gpu.sh
tmux attach -t embedding-gpu
~~~

El script arranca la sesión tmux solo si hace falta; no mata sesiones existentes. El
servidor escucha **solo** en 127.0.0.1:8765. Para ejecutarlo en primer plano durante
desarrollo:

~~~bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python gpu_server.py
~~~

En el equipo local, si el túnel no está activo:

~~~bash
ssh -o ExitOnForwardFailure=yes -N -L 127.0.0.1:18765:127.0.0.1:8765 kratos
~~~

Prueba desde el equipo local:

~~~bash
curl http://127.0.0.1:18765/health
curl -sS http://127.0.0.1:18765/embed \
  -H 'Content-Type: application/json' \
  -d '{"items":[{"text":"un perro"},{"text":"una pizza"}]}'
~~~

prueba.py ejecuta una prueba directa del modelo:

~~~bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python prueba.py
~~~

## API

GET /health devuelve estado, dispositivo, modelo, dimensión y si el modelo está cargado.
POST /embed acepta:

~~~json
{
  "items": [
    {"text": "un perro"},
    {"image_base64": "<JPEG, PNG o WebP en base64>"},
    {"text": "un perro", "image_base64": "<imagen en base64>"}
  ],
  "instruction": "Retrieve images of animals"
}
~~~

La respuesta contiene vectors en el mismo orden que items y dimension: 2048.
Los vectores ya están normalizados; el producto escalar entre dos vectores es su
similitud coseno. image_base64 acepta base64 puro o un data URL. Máximo: 16 elementos,
25 MiB por solicitud, 8 MiB y 20 megapíxeles por imagen. Los errores tienen
error y detail en JSON. El servidor procesa una solicitud a la vez para evitar
cargas concurrentes en la GPU.

POST /inspect muestra un resumen del recorrido del **último token** por las 28 capas
del decoder, durante una inferencia normal. Usa el mismo modelo ya cargado:

~~~bash
curl -sS http://127.0.0.1:18765/inspect \
  -H 'Content-Type: application/json' \
  -d '{"item":{"text":"un perro en la playa"}}'
~~~

También acepta `{"item":{"image_base64":"<imagen>"}}` e `instruction` opcional al
nivel superior. Se permite **un solo elemento**, texto de hasta 1000 caracteres o
imagen de hasta 1 MiB y 512 píxeles por lado; no ambos juntos. La respuesta contiene
`model`, `dimension`, `modality`, `input_token_count` y `layers`. Cada capa incluye:

- `layer`: número del decoder, de 1 a 28.
- `rms`: raíz del promedio de los cuadrados de las 2048 coordenadas.
- `mean` y `std`: media y desviación estándar de esas coordenadas.
- `delta_cos`: coseno directo entre el vector del último token en esta capa y en la
  capa anterior; es `null` en la primera. Un valor menor indica mayor giro.

La imagen primero pasa por el codificador visual y después influye en el decoder.
Esta inspección mide el recorrido **en el decoder**, no las capas internas del
codificador visual. Los hooks copian solo el vector del último token por capa para
calcular estadísticas y no envían activaciones completas por la red. Son señales
descriptivas de una inferencia; no representan un entrenamiento ni atribución causal.

## Recursos compartidos y reproducción

embed.py, gpu_server.py, prueba.py, start-gpu.sh y este README son archivos editables
dentro de este workspace. .venv, models/Qwen3-VL-Embedding-2B y samples son
**enlaces simbólicos** a recursos existentes en
/datos/workspaces/personal/tinkerers-embeddings; evitan duplicar unos 12 GB.
Editar este código no modifica tinkerers, pero instalar paquetes a través de .venv
sí modificaría su entorno compartido.

Para aislar este workspace por completo, sustituye esos enlaces por recursos propios.
Las versiones que han funcionado aquí son Python 3.12.13, PyTorch 2.11.0+cu128,
TorchVision 0.26.0+cu128 y las dependencias fijadas en requirements.txt. Por ejemplo,
tras retirar solo los enlaces simbólicos .venv y models/Qwen3-VL-Embedding-2B:

~~~bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen3-VL-Embedding-2B', local_dir='models/Qwen3-VL-Embedding-2B')"
~~~
