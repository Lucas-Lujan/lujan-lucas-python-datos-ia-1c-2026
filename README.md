# ⚖️ 𝗝𝘂𝘀𝗜𝗔: 𝐴𝑛𝑎𝑙𝑖𝑧𝑎𝑑𝑜𝑟 𝐽𝑢𝑟𝜄́𝑑𝑖𝑐𝑜 (RAG Local y Reflexivo)

Curso Python, Datos e Ingeniería de IA Aplicada — UTN Rosario

════════════════════════════════════════════════════════════════

# ¿Qué es JusIA?

JusIA es un asistente digital enfocado en procesar y analizar sentencias, extrayendo resúmenes estructurados (partes del caso, hechos, pruebas) y fichas técnicas en JSON mediante un sistema RAG local, privado y gratuito basado en Ollama.

 JusIA corre por completo dentro de la computadora del usuario, garantizando confidencialidad absoluta mediante la orquestación local de **Ollama** (`gemma2:2b`) y un motor de búsqueda semántica basado en matemáticas vectoriales.

────────────────

 # Características

1. **Garantía de Confidencialidad**: Procesamiento offline que mantiene los expedientes bajo llave dentro de tu memoria local.
2. **Buscador Semántico Local**: Implementación nativa de TF-IDF + Similitud Coseno para recuperar fragmentos del PDF por su significado profundo y no por coincidencia de palabras exactas.
3. **Doble Escudo de Control (RAG + Reflexión)**: 
   * **Capa 1 (Generador)**: Consulta a la IA para redactar un borrador estructurado en formato JSON y un resumen de lectura rápida.
   * **Capa 2 (Auditor)**: Un agente corrector evalúa el borrador frente a las evidencias originales mediante una rúbrica rigurosa de calidad, evitando alucinaciones de nombres, montos o fechas.
4. **Heurísticas Judiciales**: El buscador filtra y restringe semánticamente la información según la foja del expediente (ej: las partes se extraen estrictamente de la Foja 1; la decisión final, de las últimas páginas), bloqueando colisiones de palabras clave.
5. **Garantía de Grounding**: Cada dato de la ficha y cada línea de las evidencias incluye la referencia a su número de página real en el fallo original para verificación del abogado.
6 **Interfaz Gradio Blocks**: Panel interactivo moderno (`Soft()`) que divide la pantalla en una columna de entrada de documentos y un visor derecho con tres pestañas: Resumen Ejecutivo, Ficha Técnica JSON y Evidencias de Respaldo.

────────────────

# 🏗️ Estructura del Repositorio

Tu carpeta de trabajo debe verse organizada de la siguiente manera:

```text
lujan-lucas-ia-2026/
├── fallos_prueba/             # PDFs reales de sentencias (ej: Video Club Dreams, Aquino)
│   ├── Aquino-Isacio-Argentina-2004.pdf
│   └── Video-Club-Dreams-c.-Instituto-Nacional...pdf
├── .env.ejemplo               # Plantilla de variables de entorno
├── .gitignore                 # Archivo para evitar subir carpetas pesadas (.venv, .env)
├── app.py                     # Código principal unificado de la aplicación (Gradio + Ollama + RAG)
├── Dockerfile                 # Receta de empaquetado para portabilidad absoluta
└── requirements.txt           # Librerías de Python requeridas para la ejecución
```

────────────────

# Instalación y Ejecución Rápida

# Prerrequisitos de Sistema

1. Tener instalado **Ollama Desktop** ([Descargar de ollama.com](https://ollama.com/download)).
2. Tener el modelo de Google descargado localmente (ejecuta esto en tu consola física):
   ```bash
   ollama pull gemma2:2b
   ```
3. *(Opcional)* Tener **Docker Desktop** activo si prefieres correrlo en un contenedor.

────────────────

# Opción A: Ejecución en Entorno Virtual (Local)

1. Abre tu terminal en la raíz del proyecto `lujan-lucas-ia-2026/`.
2. Crea y activa el entorno virtual de Python usando `uv` (o `venv` estándar):
   ```bash
   # Con uv (Recomendado):
   uv venv .venv --python 3.12
   source .venv/bin/activate      # En Linux/macOS
   .venv\Scripts\Activate.ps1     # En Windows PowerShell
   ```
3. Instala las dependencias necesarias:
   ```bash
   uv pip install -r requirements.txt
   ```
4. Ejecuta el archivo principal:
   ```bash
   python app.py
   ```
5. Abre en tu navegador de internet:  **[http://localhost:7860](http://localhost:7860)**

────────────────

# Opción B: Ejecución Portátil con Docker

Docker empaqueta el backend y sus dependencias de forma idéntica, evitando conflictos de librerías en tu sistema operativo:

1. Inicia **Docker Desktop**
2. Construye la imagen de tu contenedor localmente:
   
   docker build -t jusia-app .
   
3. Corre el contenedor en el puerto `7860`:
   
   docker run -p 7860:7860 jusia-app
   
4. Abre en tu navegador de internet:  **[http://localhost:7860](http://localhost:7860)**

────────────────

##  Arquitectura del Pipeline de Datos (RAG + Reflexión) (¿Cómo funciona?)

El procesamiento de sentencias se ejecuta de forma estructurada en un flujo de 4 pasos principales:

📥 Subida PDF (Subida de la sentencia en formato PDF)
     │
     ▼
✂️ Capa de Chunking (Segmentación en tarjetas de 1000 caracteres)
     │
     ▼
🔍 Recuperación Semántica (TF-IDF + Coseno restringido por heurística de fojas)
     │
     ▼
🤖 Capa 1: gemma2:2b (Generación del primer Borrador en formato JSON)
     │
     ▼
🛡️ Capa 2: gemma2:2b (Auditoría de Calidad contra rúbrica estricta)
     │
     ▼
📋 Gradio Blocks (Visualización interactiva y pestaña de Evidencias de Grounding)

────────────────

# Tecnologías y Librerías Utilizadas

* **`gradio`**: Para el desarrollo de la interfaz de usuario interactiva por pestañas.
* **`ollama`**: Cliente de conexión local con el modelo LLM offline.
* **`pypdf`**: Extractor robusto de texto plano página por página de los archivos PDF.
* **`scikit-learn`**: Motor matemático local para vectorización léxica (**TfidfVectorizer**) y cálculo de cercanía (**cosine_similarity**).
* **`numpy`**: Procesamiento veloz de arrays y ordenamiento de scores de similitud.
* **`python-dotenv`**: Carga segura de configuraciones del entorno local.

---
