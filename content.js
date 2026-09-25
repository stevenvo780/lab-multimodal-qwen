/* Contenido didáctico local. No se descarga información de un servicio externo. */
window.ATLAS_CONTENT = {
  categories: [
    { id: "todos", label: "Todos" },
    { id: "inputs", label: "Entradas" },
    { id: "modelo", label: "Modelo" },
    { id: "comparar", label: "Comparar" },
    { id: "aplicar", label: "Aplicar" }
  ],
  tomographyViews: {
    capas: {
      title: "Capas internas",
      description: "La inspección sigue estadísticas de estados intermedios para una entrada a lo largo de las 28 capas del decoder. Describe cambios; no demuestra qué los causa.",
      question: "¿Qué cambia entre capas y qué permanece estable para el mismo ejemplo?",
      alt: "Esquema de cambios de representación a lo largo de capas internas"
    },
    mapa: {
      title: "Mapa de embeddings finales",
      description: "PCA proyecta embeddings reales de varios ejemplos en dos dimensiones para ver grupos y vecinos. La proyección puede deformar distancias del espacio original.",
      question: "¿Se agrupan por etiqueta, por modalidad o por otra característica?",
      alt: "Mapa esquemático de ejemplos representados como puntos"
    },
    perdida: {
      title: "Curva de pérdida de la sonda",
      description: "La curva muestra cómo cambia el error de entrenamiento de la sonda. Una caída no garantiza que acierte en ejemplos nuevos.",
      question: "¿La sonda mejora también en ejemplos que no usó para aprender?",
      alt: "Curva esquemática de pérdida de entrenamiento descendente"
    },
    prediccion: {
      title: "Predicciones en casos nuevos",
      description: "La sonda ya entrenada recibe embeddings de entradas nuevas y estima etiquetas. Revisar aciertos y errores completa la inspección.",
      question: "¿Qué casos falla y qué tienen en común?",
      alt: "Esquema de entrada nueva, embedding y predicción de una sonda"
    }
  },
  companyTypeNames: {
    bank_statement: "Extracto bancario",
    chart: "Gráfico",
    financial_table: "Tabla financiera",
    lease: "Contrato de arriendo",
    medical: "Documento médico",
    payslip: "Nómina",
    photo_receipt: "Recibo fotografiado",
    receipt: "Recibo escaneado",
    shipping: "Documento de envío",
    tax_form: "Formulario fiscal"
  },
  concepts: [
    {
      id: "multimodal", title: "¿Qué significa multimodal?", category: "inputs", icon: "▧", step: "01 · Entradas",
      short: "Un mismo experimento combina información que llega como texto e imagen.",
      summary: "Texto e imagen son modalidades distintas: una usa palabras y la otra píxeles. Un modelo multimodal puede procesar ambas para comparar su contenido.",
      why: "Permite preguntar con una frase y encontrar imágenes relacionadas, o examinar si una descripción corresponde a una imagen.",
      test: "En la matriz, escribe dos frases: una que describa claramente una imagen y otra que no. ¿Cuál queda más cerca?",
      caution: "Tener dos modalidades no garantiza que el modelo entienda todos los detalles de una escena.",
      example: "«Un gato sobre un sofá» ↔ fotografía de un gato en un sofá."
    },
    {
      id: "preparacion", title: "Preparación de entradas", category: "inputs", icon: "◫", step: "02 · Entradas",
      short: "Antes del modelo, cada entrada se convierte al formato que espera su procesador.",
      summary: "El texto se divide en unidades de entrada y la imagen se transforma en datos visuales procesables. La forma exacta depende del modelo y su procesador.",
      why: "Cambios al recorte de una imagen o a la redacción de una frase pueden cambiar el vector obtenido.",
      test: "Prueba una frase corta y otra más descriptiva sobre la misma imagen. Observa si cambia la similitud.",
      caution: "No compares números de ejecuciones que usan modelos o preparación de datos diferentes como si fueran la misma escala.",
      example: "«perro» y «perro negro corriendo en un parque» son entradas distintas."
    },
    {
      id: "encoder", title: "Codificador visual y textual", category: "modelo", icon: "✳", step: "03 · Modelo",
      short: "Qwen3-VL Embedding transforma entradas en representaciones numéricas.",
      summary: "Un codificador recibe una entrada y devuelve un vector. En este laboratorio usamos Qwen3-VL Embedding para representar texto e imágenes.",
      why: "Es el paso que hace posible comparar datos que originalmente tienen formas distintas.",
      test: "Compara una misma frase dos veces sin cambiar su contenido. Después cambia una palabra importante y mira qué se mueve.",
      caution: "Este modelo de embeddings produce vectores; la bitácora sirve para conversar sobre ideas, pero no simula un chat generativo.",
      example: "codificador(«una bicicleta») → vector de números"
    },
    {
      id: "embedding", title: "Embedding o vector", category: "modelo", icon: "[ ]", step: "04 · Modelo",
      short: "Una lista de números ubica cada entrada en un espacio de muchas dimensiones.",
      summary: "Un embedding es una representación vectorial. Cada número es una coordenada; juntos, los números permiten medir relaciones entre entradas.",
      why: "Es la unidad básica de la matriz, la búsqueda, la aritmética y las proyecciones que verás en Streamlit.",
      test: "En el laboratorio, selecciona dos entradas parecidas y otras diferentes. Observa cómo cambian sus puntuaciones, no una sola coordenada aislada.",
      caution: "Una coordenada individual casi nunca tiene una interpretación sencilla como «cantidad de gatos».",
      example: "Esquema de juguete: [0,8; 0,2; −0,1]. El vector real puede tener muchas más dimensiones."
    },
    {
      id: "espacio", title: "Espacio compartido", category: "modelo", icon: "⌁", step: "05 · Modelo",
      short: "Texto e imagen se representan de manera que sus vectores se puedan comparar.",
      summary: "Las representaciones de ambas modalidades están alineadas: una frase y una imagen relacionadas pueden quedar próximas en el espacio aprendido.",
      why: "Gracias a ese espacio común, una búsqueda textual puede devolver imágenes y una matriz puede cruzar las dos modalidades.",
      test: "Describe una misma foto de dos formas y observa cuáles palabras acercan o alejan la consulta.",
      caution: "El mapa en dos dimensiones de esta web es un dibujo. El espacio real tiene más dimensiones y puede contener errores o sesgos.",
      example: "«un perro» y la foto de un perro deberían competir como vecinos cercanos."
    },
    {
      id: "normalizacion", title: "Norma y normalización", category: "comparar", icon: "↗", step: "06 · Comparar",
      short: "La longitud del vector y su dirección cuentan cosas distintas al calcular similitud.",
      summary: "La norma indica la longitud de un vector. Normalizarlo divide sus coordenadas por esa norma para dejar longitud uno, conservando la dirección.",
      why: "Cuando los vectores están normalizados, el producto punto coincide con la similitud coseno.",
      test: "En el explorador de esta web, cambia solo la longitud de B. ¿Se modifica el coseno?",
      caution: "No asumas que el servicio ya normaliza los vectores: hay que revisar la implementación antes de interpretar el producto punto.",
      example: "[2, 0] y [1, 0] apuntan igual; su coseno es 1 aunque sus longitudes difieran."
    },
    {
      id: "coseno", title: "Similitud coseno", category: "comparar", icon: "∠", step: "07 · Comparar",
      short: "Mide cuánto se parecen las direcciones de dos vectores.",
      summary: "La similitud coseno es el producto punto dividido por el producto de las normas. Vale entre −1 y 1; 1 indica la misma dirección, 0 direcciones perpendiculares.",
      why: "Ordena los resultados de una búsqueda y da un valor a cada cruce de la matriz texto↔imagen.",
      test: "Mueve el ángulo en el explorador. Comprueba qué sucede cerca de 0°, 90° y 180°.",
      caution: "Un valor alto indica cercanía según este modelo; no es una probabilidad ni una garantía de verdad.",
      example: "cos(A, B) = (A · B) / (||A|| × ||B||)"
    },
    {
      id: "matriz", title: "Matriz texto ↔ imagen", category: "comparar", icon: "▦", step: "08 · Comparar",
      short: "Una tabla muestra la similitud de cada frase con cada imagen.",
      summary: "Las filas pueden ser textos y las columnas imágenes. Cada celda guarda la puntuación para ese par. Si los pares alineados son buenos, suele aparecer una diagonal de valores altos.",
      why: "Ayuda a ver de golpe aciertos, confusiones y casos donde varias descripciones sirven para la misma imagen.",
      test: "Pon dos imágenes parecidas y dos descripciones. Busca una celda fuera de la diagonal con valor alto e intenta explicarla.",
      caution: "La diagonal solo tiene sentido si ordenaste filas y columnas como pares correspondientes.",
      example: "Filas: «gato», «bicicleta». Columnas: foto gato, foto bicicleta."
    },
    {
      id: "busqueda", title: "Búsqueda semántica", category: "aplicar", icon: "⌕", step: "09 · Aplicar",
      short: "Una consulta recupera los vectores más cercanos de una colección.",
      summary: "Primero representas la consulta y los elementos de la colección. Después calculas similitudes y ordenas los candidatos de mayor a menor.",
      why: "Es una aplicación directa del espacio compartido: puedes consultar con texto y recuperar imágenes.",
      test: "Busca una escena con una frase breve y luego añade detalles. ¿Cambia el primer resultado?",
      caution: "El resultado más cercano siempre existe aunque ninguno sea realmente pertinente. Hay que inspeccionarlo.",
      example: "Consulta: «persona en bicicleta» → fotos ordenadas por similitud."
    },
    {
      id: "aritmetica", title: "Aritmética vectorial", category: "aplicar", icon: "±", step: "10 · Aplicar",
      short: "Sumar y restar vectores sirve para explorar desplazamientos en el espacio.",
      summary: "Operaciones como A − B + C producen un nuevo punto. Buscar sus vecinos ayuda a investigar si cierto cambio de significado se refleja en el espacio.",
      why: "Es una forma exploratoria de preguntar cómo se organizan las relaciones aprendidas.",
      test: "Elige A y B que se diferencien en un rasgo, añade C y observa si aparece ese rasgo entre los vecinos.",
      caution: "No toda analogía funciona, y un vecino cercano no demuestra una regla general.",
      example: "Vector A − vector B + vector C → mirar sus vecinos; no predecir el resultado de antemano."
    },
    {
      id: "proyeccion", title: "Proyección PCA y UMAP", category: "aplicar", icon: "◈", step: "11 · Aplicar",
      short: "Reduce un espacio de muchas dimensiones para poder dibujarlo.",
      summary: "PCA y UMAP generan vistas en dos dimensiones. Son herramientas de visualización y cada método conserva aspectos distintos de los datos.",
      why: "Te permiten inspeccionar agrupaciones y casos atípicos, sin tener que mirar cientos o miles de coordenadas.",
      test: "Proyecta el mismo conjunto con PCA y UMAP. Busca vecinos que se mantienen juntos y otros que cambian.",
      caution: "Una distancia en la gráfica 2D no sustituye la similitud calculada en el espacio original.",
      example: "Un conjunto de imágenes de animales podría formar grupos visibles, según los datos y la proyección."
    },
    {
      id: "evaluacion", title: "Evaluar lo que observas", category: "aplicar", icon: "✓", step: "12 · Aplicar",
      short: "Convierte una impresión visual en una prueba que se pueda repetir.",
      summary: "Define qué esperabas, cambia una sola cosa y anota el resultado. Repite con ejemplos distintos para ver si la observación se sostiene.",
      why: "Un buen ejemplo puede impresionar, pero varios controles muestran mejor dónde ayuda el modelo y dónde falla.",
      test: "Para cada hallazgo, crea un caso parecido y un caso de control. Guarda la consulta, la imagen y la posición de los resultados.",
      caution: "Una proyección vistosa o una coincidencia aislada no es evidencia suficiente de calidad general.",
      example: "Hipótesis → ejemplo positivo → control → resultado → pregunta siguiente."
    }
  ],
  practices: [
    {
      id: "matriz", number: "01", title: "Matriz texto ↔ imagen", subtitle: "Pestaña «Matriz» · compara todos los pares",
      steps: ["Elige dos imágenes claramente distintas y descríbelas.", "Mira las celdas de cada par. Cambia una descripción por otra ambigua.", "Anota qué celda subió o bajó y por qué crees que ocurrió."],
      question: "¿Los pares correctos destacan de los distractores?",
      concept: "matriz"
    },
    {
      id: "busqueda", number: "02", title: "Búsqueda semántica", subtitle: "Pestaña «Búsqueda» · consulta una colección",
      steps: ["Busca una imagen con una frase corta.", "Agrega un atributo concreto: color, acción o contexto.", "Compara el orden de los primeros resultados y registra el cambio."],
      question: "¿Qué parte de la frase parece mover más el ranking?",
      concept: "busqueda"
    },
    {
      id: "aritmetica", number: "03", title: "Aritmética vectorial", subtitle: "Pestaña «Aritmética» · prueba A − B + C",
      steps: ["Elige tres ejemplos cuyo cambio puedas explicar.", "Haz la operación y revisa los vecinos del vector resultante.", "Prueba otro trío para ver si la relación se repite."],
      question: "¿Aparece el rasgo esperado o solo una coincidencia?",
      concept: "aritmetica"
    },
    {
      id: "proyeccion", number: "04", title: "PCA / UMAP", subtitle: "Pestaña «Proyección» · mira el mapa",
      steps: ["Proyecta un mismo conjunto con PCA.", "Cambia a UMAP sin cambiar los ejemplos.", "Identifica qué agrupaciones se mantienen y cuáles cambian."],
      question: "¿Qué conclusión depende demasiado del dibujo 2D?",
      concept: "proyeccion"
    },
    {
      id: "tomografia", number: "05", title: "Capas, mapa y sonda", subtitle: "Módulo «Tomografía» · entrenamiento y predicción",
      steps: ["Inspecciona las capas internas para un texto y una imagen.", "Mira el mapa PCA y entrena la sonda con ejemplos etiquetados.", "Compara la curva de pérdida con predicciones de casos nuevos."],
      question: "¿Qué observación describe una representación y cuál demuestra utilidad en ejemplos nuevos?",
      concept: "tomografia"
    }
  ],
  glossary: [
    { term: "Vector", definition: "Lista ordenada de números que representa una entrada." },
    { term: "Dimensión", definition: "Una coordenada del vector. Un embedding real suele tener muchas." },
    { term: "Norma", definition: "Longitud de un vector, calculada a partir de sus coordenadas." },
    { term: "Producto punto", definition: "Suma de productos de coordenadas correspondientes de dos vectores." },
    { term: "Coseno", definition: "Comparación de direcciones, independiente de la longitud de los vectores." },
    { term: "Vecino próximo", definition: "Elemento con una similitud alta respecto a una consulta." },
    { term: "Ranking", definition: "Lista de resultados ordenada por puntuación." },
    { term: "Proyección", definition: "Representación de datos de alta dimensión en dos o tres ejes." },
    { term: "Control", definition: "Ejemplo que ayuda a comprobar si una observación se mantiene." },
    { term: "Sonda", definition: "Clasificador pequeño entrenado sobre embeddings de un modelo congelado." },
    { term: "InfoNCE", definition: "Pérdida contrastiva que favorece pares relacionados frente a distractores durante entrenamiento." },
    { term: "Tomografía", definition: "Metáfora para inspeccionar capas internas y relaciones entre embeddings con varias vistas." }
  ]
};
