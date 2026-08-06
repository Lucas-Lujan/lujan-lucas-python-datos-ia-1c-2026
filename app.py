import os
import json
import time
from pathlib import Path
import numpy as np
from pypdf import PdfReader
from dotenv import load_dotenv
import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ollama

# ==============================================================================
# SEGMENTO 1: CARGA DE CONFIGURACIONES Y VARIABLES DE ENTORNO
# Explicación simple: Aquí cargamos las herramientas básicas del sistema y leemos
# de forma segura nuestro archivo oculto '.env' para cargar configuraciones sin
# escribir datos sensibles directo en el código.
# ==============================================================================

load_dotenv()
# ── CONFIGURACIÓN DEL MOTOR LOCAL DE OLLAMA ──
# Explicación simple: Para hablar con Ollama desde adentro de un contenedor Docker
# usamos la ruta 'host.docker.internal' (el puente hacia tu computadora). Si corremos
# la aplicación directamente con Python, usamos 'localhost'. El sistema prueba ambas.
OLLAMA_HOST_DOCKER = "http://host.docker.internal:11434"
OLLAMA_HOST_LOCAL = "http://localhost:11434"

# Modelo 'gemma2:2b' ya que es liviano y potente
MODELO_OLLAMA = "gemma2:2b" 

# Inicializamos el cliente de Ollama con estrategia de redundancia (Docker -> Local)
def obtener_cliente_ollama():
    """Intenta conectarse a Ollama en el puente de Docker. Si falla, cae al localhost estándar."""
    try:
        # Intento 1: Conexión mediante el puente de red de Docker
        cliente = ollama.Client(host=OLLAMA_HOST_DOCKER)
        cliente.list()
        print(f"✓ Conectado a Ollama en el Host de Docker: {OLLAMA_HOST_DOCKER}")
        return cliente
    except Exception:
        try:
            # Intento 2: Conexión local estándar en tu computadora física
            cliente = ollama.Client(host=OLLAMA_HOST_LOCAL)
            cliente.list()
            print(f"✓ Conectado a Ollama Local: {OLLAMA_HOST_LOCAL}")
            return cliente
        except Exception as e:
            print(f"⚠ No se pudo conectar a Ollama en ninguna de las rutas. Error: {e}")
            return None

# Instanciamos el cliente para que quede listo para ser usado por el sistema
ollama_client = obtener_cliente_ollama()

# ── FUNCIÓN AUXILIAR: CONSULTA OLLAMA CON REINTENTOS ──
def consultar_ollama_local(prompt, system_prompt=None, response_json=False):
    """Realiza la consulta al cliente de Ollama configurado."""
    global ollama_client
    if ollama_client is None:
        ollama_client = obtener_cliente_ollama()
        if ollama_client is None:
            raise RuntimeError("Ollama no se encuentra ejecutándose en tu máquina. Por favor inicia Ollama Desktop.")

    mensajes = []
    if system_prompt:
        mensajes.append({"role": "system", "content": system_prompt})
    mensajes.append({"role": "user", "content": prompt})

    # Configuramos el formato JSON si es requerido
    options = {"temperature": 0.1}
    format_type = "json" if response_json else ""

    respuesta = ollama_client.chat(
        model=MODELO_OLLAMA,
        messages=mensajes,
        format=format_type,
        options=options
    )
    return respuesta.message.content

# ==============================================================================
# SEGMENTO 2: LIMPIEZA LÉXICA Y MOTOR VECTORIAL DE BÚSQUEDA (TF-IDF + COSENO) 
# TF (Frecuencia de término): Cuenta cuántas veces se repite una palabra en un texto propio. Si se repite mucho, su valor de TF sube.
# IDF (Frecuencia inversa de documento): Mide si una palabra es rara o común en todos los textos analizados. Las palabras que salen en casi todos lados (como "el", "la", "de") obtienen un valor muy bajo.
# Explicación simple: Antes de buscar, filtramos palabras de relleno (stopwords).
# Usamos las matemáticas para medir el peso de cada término
# y la Similitud Coseno para encontrar la ficha de texto que mejor responde a cada pregunta.
# ==============================================================================

# ── COINCIDENCIA LÉXICA LOCAL (MOTOR VECTORIAL TF-IDF) ──
STOPWORDS = {"de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por", "un", "para", "con", "no", "una", "su", "al", "lo", "como", "más", "pero", "sus", "este", "o", "este", "ese", "es", "esta", "son", "entre", "cuando", "muy", "sin", "sobre", "ser", "sus", "también", "me", "nos", "lo", "yo", "mi"}

def similitud_lexica_jaccard(texto_a, texto_b):
    """Calcula la similitud de Jaccard entre dos textos eliminando signos de puntuación y stopwords."""
    def obtener_palabras(texto):
        palabras = str(texto).lower().split()
        palabras_limpias = set()
        for p in palabras:
            p_limpia = "".join(c for p_part in p.split("-") for c in p_part if c.isalnum())
            if p_limpia and p_limpia not in STOPWORDS:
                palabras_limpias.add(p_limpia)
        return palabras_limpias

    set_a = obtener_palabras(texto_a)
    set_b = obtener_palabras(texto_b)
    
    if not set_a or not set_b:
        return 0.0
        
    interseccion = set_a.intersection(set_b)
    union = set_a.union(set_b)
    return len(interseccion) / len(union)


class AlmacenVectorialLocal:
    """Almacén vectorial ligero local de NumPy basado en TF-IDF (Clase 6). 
    Perfecto para correr offline sin gastar cuota."""
    def __init__(self):
        self.chunks = []      # Lista de fragmentos de texto
        self.metadatas = []   # Lista de metadatos (ej: {"pagina": 3})
        self.vectorizador_tfidf = None
        self.matriz_tfidf = None

    def agregar_documentos(self, chunks, metadatas):
        self.chunks.extend(chunks)
        self.metadatas.extend(metadatas)
        if self.chunks:
            self.vectorizador_tfidf = TfidfVectorizer(stop_words=list(STOPWORDS))
            self.matriz_tfidf = self.vectorizador_tfidf.fit_transform(self.chunks)
        
    def busqueda_semantica(self, query_text, k=1):
        if not self.chunks or self.vectorizador_tfidf is None:
            return []
        
        query_vec = self.vectorizador_tfidf.transform([query_text])
        similitudes = cosine_similarity(query_vec, self.matriz_tfidf)[0]
        
        resultados = []
        for idx, score in enumerate(similitudes):
            resultados.append({
                "texto": self.chunks[idx],
                "metadata": self.metadatas[idx],
                "score": float(score)
            })
        resultados.sort(key=lambda x: x["score"], reverse=True)
        return resultados[:k]


# ==============================================================================
# SEGMENTO 3: LECTURA DE PDF Y TROCEADO DE TEXTO (CHUNKING CON SOLAPAMIENTO)
# Explicación simple: Esta es la 'tijera digital'. Lee el PDF página por página,
# corta el texto en tarjetas de estudio (chunks) de 1000 letras (caracteres)
# y solapa las últimas 150 para que ninguna frase importante quede cortada al medio.
# ==============================================================================


# ── UTILIDADES DE PROCESAMIENTO Y CHUNKING ──
def segmentar_texto(texto, chunk_size=1000, chunk_overlap=150):
    chunks = []
    inicio = 0
    while inicio < len(texto):
        fin = inicio + chunk_size
        chunk = texto[inicio:fin]
        chunks.append(chunk)
        # Retrocedemos el overlap para la siguiente tarjeta
        inicio += chunk_size - chunk_overlap
    return chunks

def procesar_pdf(ruta_pdf):
    lector = PdfReader(ruta_pdf)
    chunks_totales = []
    metadatas_totales = []
    
    for i, pagina in enumerate(lector.pages):
        numero_pagina = i + 1
        texto_pagina = pagina.extract_text() or ""
        if not texto_pagina.strip():
            continue

        # Segmentamos cada hoja conservando siempre en sus metadatos la página de origen (Grounding)    
        chunks_pagina = segmentar_texto(texto_pagina, chunk_size=1000, chunk_overlap=150)
        for chunk in chunks_pagina:
            chunks_totales.append(chunk)
            metadatas_totales.append({"pagina": numero_pagina})
    # Creamos e indexamos la base vectorial nativa de nuestra app        
    almacen = AlmacenVectorialLocal()
    if chunks_totales:
        almacen.agregar_documentos(chunks_totales, metadatas_totales)
    return almacen, len(lector.pages)

# ==============================================================================
# SEGMENTO 4: HEURÍSTICAS DE BÚSQUEDA JUDICIAL (EVITAR COLISIÓN DE PALABRAS CLAVE)
# Explicación simple: El derecho es muy estructurado. 
# 1. Los nombres de las partes están en Foja 1 (Página 1).
# 2. Los hechos que originan el pleito están en las primeras páginas (1 a 5).
# 3. La resolución final ('Por ello, se resuelve...') se encuentra en las últimas hojas.
# Para evitar confundirnos con discusiones del medio, este bloque busca en las fojas exactas.
# ==============================================================================

def recuperar_contexto_y_evidencias(almacen):
    """Divide las evidencias de forma didáctica en las 4 categorías clave sugeridas por el usuario
    aplicando heurísticas avanzadas de dominio judicial para evitar la deriva semántica (Keyword Collision)."""
    categorias_legales = [
        {
            "titulo": "Evidencia 1: Identificación de Actores y Demandados",
            "query": "quiénes son las partes actora demandante demandada acusado nombres de personas fisicas o juridicas caratula contra c/",
            "proposito": "Determinar los sujetos involucrados en el conflicto (quién demanda y a quién)."
        },
        {
            "titulo": "Evidencia 2: Hechos y Reclamaciones (Objeto del Conflicto)",
            "query": "hechos del caso demanda conflicto reclamo objeto del proceso controversia daño inconstitucionalidad de decretos leyes resolucion",
            "proposito": "Comprender el contexto fáctico, los sucesos y qué solicita la parte actora."
        },
        {
            "titulo": "Evidencia 3: Pruebas Presentadas por las Partes",
            "query": "pruebas documental testimonial pericial testigos actas contratos facturas peritos documentacion informes peritajes",
            "proposito": "Analizar los elementos de prueba aportados por el actor y el demandado para sustentar sus posturas."
        },
        {
            "titulo": "Evidencia 4: Sentencia o Decisión Final",
            "query": "juez tribunal sala sentencia resolucion decision condena absolucion monto multa prision pena resuelve por ello declarase admitir revocar",
            "proposito": "Determinar el veredicto del juez o tribunal, y las consecuencias jurídicas finales decretadas."
        }
    ]
    
    total_paginas = max([m.get("pagina", 1) for m in almacen.metadatas]) if almacen.metadatas else 1
    
    def buscar_en_rango_paginas(query_text, pag_inicio, pag_fin, k=1):
        # Filtramos los chunks del almacén que se ubican en el rango de páginas (inclusive)
        chunks_filtrados = []
        metadatas_filtrados = []
        
        for idx, meta in enumerate(almacen.metadatas):
            pag = meta.get("pagina", 1)
            if pag_inicio <= pag <= pag_fin:
                chunks_filtrados.append(almacen.chunks[idx])
                metadatas_filtrados.append(meta)
                
        if not chunks_filtrados:
            return almacen.busqueda_semantica(query_text, k)
            
        # Iniciamos búsqueda TF-IDF temporal con los fragmentos de las fojas filtradas
        if almacen.vectorizador_tfidf is not None:
            try:
                vectorizador_temp = TfidfVectorizer(stop_words=list(STOPWORDS))
                matriz_temp = vectorizador_temp.fit_transform(chunks_filtrados)
                query_vec = vectorizador_temp.transform([query_text])
                similitudes = cosine_similarity(query_vec, matriz_temp)[0]
                
                resultados = []
                for idx, score in enumerate(similitudes):
                    resultados.append({
                        "texto": chunks_filtrados[idx],
                        "metadata": metadatas_filtrados[idx],
                        "score": float(score)
                    })
                resultados.sort(key=lambda x: x["score"], reverse=True)
                return resultados[:k]
            except Exception as e:
                print(f"Error en búsqueda TF-IDF local restringida: {e}")
                
        # Fallback manual simple Jaccard si falla la matriz
        resultados = []
        for idx, chunk in enumerate(chunks_filtrados):
            similitud = similitud_lexica_jaccard(query_text, chunk)
            resultados.append({
                "texto": chunk,
                "metadata": metadatas_filtrados[idx],
                "score": similitud
            })
        resultados.sort(key=lambda x: x["score"], reverse=True)
        return resultados[:k]

    evidencias_lista = []
    bloques_contexto = []
    
    for idx, cat in enumerate(categorias_legales, 1):
        if idx == 1:
            # HEURÍSTICA 1: La carátula y los nombres de las partes se extraen estrictamente de la Página 1 (foja 1).
            resultados = buscar_en_rango_paginas(cat["query"], 1, 1, k=1)
        elif idx == 2:
            # HEURÍSTICA 2: Los hechos y el objeto de la demanda se extraen de las Páginas 1 a 5 (fojas iniciales).
            # Esto evita "keyword collision" con debates procedimentales complejos de páginas lejanas (como la página 44).
            resultados = buscar_en_rango_paginas(cat["query"], 1, 5, k=1)
        elif idx == 4:
            # HEURÍSTICA 4: La resolución de la Corte ("Por ello, se resuelve...") se extrae de las últimas fojas del PDF.
            # Esto evita traer resúmenes de decisiones anteriores o de cámaras de apelación descritos a mitad de la causa.
            pag_inicio_sentencia = max(1, total_paginas - 4)
            resultados = buscar_en_rango_paginas(cat["query"], pag_inicio_sentencia, total_paginas, k=1)
        else:
            # Para Pruebas (Evidencia 3), buscamos de forma abierta en todo el fallo. (No hay limitación de páginas)
            resultados = almacen.busqueda_semantica(cat["query"], k=1)
            
        if resultados:
            res = resultados[0]
            evidencias_lista.append({
                "id": idx,
                "titulo": cat["titulo"],
                "pagina": res["metadata"]["pagina"],
                "texto": res["texto"],
                "proposito": cat["proposito"]
            })
            bloques_contexto.append(f"[{cat['titulo']} - Fallo, Página {res['metadata']['pagina']}]\n{res['texto']}")
        else:
            evidencias_lista.append({
                "id": idx,
                "titulo": cat["titulo"],
                "pagina": "No detectada",
                "texto": "No se encontraron fragmentos con suficiente relevancia temática para este campo en el documento.",
                "proposito": cat["proposito"]
            })
            
    contexto_formateado = "\n\n".join(bloques_contexto)
    return contexto_formateado, evidencias_lista

# ==============================================================================
# SEGMENTO 5: EL ESCUDO DE CONTROL - PIPELINE DE RAG REFLEXIVO (DOCKER / LOCAL)
# Explicación simple: Aquí ocurren las 2 llamadas a Ollama para lograr el "Doble Escudo":
# Capa 1 (Generador): La IA actúa como "redactora" y llena los campos del JSON.
# Capa 2 (Auditor de Calidad): Una segunda llamada a la IA revisa si cometió errores,
# valida que las partes coincidan con la Página 1 y prohíbe el uso de palabras penales. (Acusado, absolver, etc)
# ==============================================================================


def analizar_y_reflexionar_local(contexto_evidencias):
    """Ejecuta el pipeline de RAG Reflexivo utilizando el modelo local de Ollama."""
    
    # ── CAPA 1: GENERACIÓN EN JSON (Llamada 1)
    prompt_borrador = f"""Sos un analista jurídico experto. Analiza el documento adjunto a partir de las evidencias provistas.
    Debes extraer información estructurada clave y redactar un resumen ejecutivo preciso.

    Responde ÚNICAMENTE con un objeto JSON estructurado que respete exactamente estas claves:
    {{
        "resumen_ejecutivo": "Un resumen redactado de forma obligatoria siguiendo esta plantilla exacta de 5 oraciones continuas: '1. [Actor] inició una demanda contra [Demandado] reclamando [Conflicto]. 2. La Cámara de Apelaciones dictó sentencia haciendo lugar a lo solicitado por la parte actora. 3. Disconforme con este pronunciamiento, [Apelante / Recurrente] interpuso un recurso ante el tribunal superior. 4. El tribunal superior declaró admisible el recurso para proceder a su análisis de fondo. 5. Finalmente, se resolvió confirmar la decisión recurrida, ratificando la victoria procesal de la parte actora.' Debes rellenar los corchetes con los datos reales del caso y omitir los corchetes en el resultado final."
        "juez_o_tribunal": "Nombre del juez/a, tribunal o sala firmante. Si no figura, escribe 'No especificado'.",
        "partes": {{
            "actor_o_demandante": "Nombre de la persona, empresa o entidad que inicia la demanda o acción de amparo original en la carátula.",
            "demandado_o_acusado": "Nombre de la contraparte demandada originalmente en la carátula."
        }},
        "fecha_resolucion": "Fecha exacta de la sentencia, fallo o acuerdo (formato dd/mm/aaaa si es posible).",
        "decision_final": "La resolución final adoptada por el tribunal o la Corte (ej: 'Hace lugar a la demanda', 'Rechaza el recurso', 'Confirma la sentencia', etc.). NUNCA utilices terminología penal como 'absolver' o 'condenar' si el juicio es administrativo, civil o laboral.",
        "monto_o_pena_impuesta": "El monto de la condena, multas o indemnización fijada. Si no aplica, escribe 'No aplica'. NUNCA uses términos penales como 'absolución' o 'prisión' en litigios civiles, laborales o administrativos."
    }}

    ⚠️ REGLAS CRÍTICAS DE COHERENCIA UNIVERSAL PARA EL RESUMEN Y LAS PARTES:
    1. REGLA ABSOLUTA DE LA CARÁTULA JURÍDICA (PÁGINA 1): La carátula oficial en Foja 1 sigue la estructura invariable \"[Sujeto A] c/ [Sujeto B]\" (o \"[Sujeto A] contra [Sujeto B]\").
       * El sujeto o entidad antes de \"c/\" o \"contra\" es STRICTAMENTE el 'actor_o_demandante' (ej: el trabajador, particular o empresa que demanda).
       * El sujeto o entidad después de \"c/\" o \"contra\" (y antes de \"s/\" o \"sobre\") es STRICTAMENTE el 'demandado_o_acusado'.
       * Esta es una ley sintáctica en el derecho argentino. Nunca inviertas estos roles.
    2. COHERENCIA DE RECURSOS Y APELACIONES: Quien ganó la causa en la instancia anterior (por ejemplo, el particular que logró la declaración de inconstitucionalidad de un impuesto en Cámara, o el trabajador que logró una condena a su favor) NO tiene motivos para apelar. Quien apela ante la Corte o instancia superior es siempre la parte que PERDIÓ en la Cámara. Por ende, el recurso de apelación o extraordinario lo interpone la parte perdedora anterior para intentar revocar ese fallo.
    3. COHERENCIA EN LA DEFENSA DE NORMAS O ACTOS: Si la contraparte es un organismo estatal o representante del Estado (como una Procuración, Fisco o Ministerio) que defiende una ley o reglamento, NUNCA redactes que dicho organismo argumentó o presentó pruebas que demuestran la 'invalidez' o 'inconstitucionalidad' de su propia norma. Quien busca demostrar la invalidez es el particular actor. El representante estatal busca defender su validez.
    4. RESOLUCIÓN DEL RECURSO: Si el tribunal superior confirma la decisión de la Cámara de Apelaciones que favorecía al actor original, esto significa que el recurso interpuesto por la contraparte perdedora fue formalmente admisible pero DESESTIMADO o RECHAZADO en cuanto al fondo, dándole la victoria definitiva a la parte actora original.
    5. NO confundas al 'recurrente' o 'apelante' (quien interpone el recurso extraordinario ante la Corte) con el 'actor_o_demandante' original de la causa.
    6. RIGOR DEL LENGUAJE JURÍDICO SEGÚN LA MATERIA: Identifica la materia del pleito. Si es derecho administrativo, civil o laboral, está estrictamente PROHIBIDO utilizar terminología penal. Bajo ninguna circunstancia uses verbos como \"absolver\", \"absuelve\" o \"acusado\", ni hables de \"delitos\" o \"penas de prisión\". Utiliza términos civilistas o laborales apropiados (ej: 'declarar inconstitucional', 'dejar sin efecto', 'hacer lugar a la indemnización', 'confirmar sentencia').

    Evidencias extraídas directamente del documento original:
    {contexto_evidencias}
"""
    
    primer_borrador_json = consultar_ollama_local(prompt_borrador, response_json=True)
    
    # ── CAPA 2: AUTO-CORRECCIÓN MEDIANTE REFLEXIÓN (Llamada 2)
    prompt_reflexion = f"""Actúa como un Auditor de Calidad Jurídica de Inteligencia Artificial. Revisa el borrador de análisis generado y corrige cualquier error, inconsistencia o alucinación, asegurándose de que cada dato esté 100% respaldado por las evidencias del documento original.

    RÚBRICA DE CALIDAD OBLIGATORIA:
    1. Resumen de 5 líneas: ¿Cumple estrictamente la plantilla de 5 oraciones continuas? La plantilla debe ser exactamente: '1. [Actor] inició una demanda contra [Demandado] reclamando [Conflicto]. 2. La Cámara de Apelaciones dictó sentencia haciendo lugar a lo solicitado por la parte actora. 3. Disconforme con este pronunciamiento, [Apelante / Recurrente] interpuso un recurso ante el tribunal superior. 4. El tribunal superior declaró admisible el recurso para proceder a su análisis de fondo. 5. Finalmente, se resolvió confirmar la decisión recurrida, ratificando la victoria procesal de la parte actora.' Verifica que los nombres de las partes estén bien colocados (ej: Video Club Dreams es el Actor y el INC es el Demandado) y corrige cualquier desvío de este formato para que la salida sea 100% prolija y uniforme.
    2. Partes: ¿Los nombres de las partes coinciden exactamente con la carátula original de la Página 1 (Evidencia 1)? Recuerda: NO confundas a la parte 'recurrente' (quien apela la sentencia anterior de cámara) con la parte actora original.
    3. Fechas y Montos: ¿La fecha y los montos coinciden exactamente con lo escrito en las evidencias? Si no figura, debe quedar 'No aplica'.
    4. Formato: ¿El formato de salida es un JSON válido que respete exactamente las claves solicitadas?
    5. Rigor de la Materia (Evitar Terminología Penal Incorrecta): En juicios administrativos, civiles o laborales, está estrictamente PROHIBIDO usar palabras penales como \"absolver\", \"absuelto\", \"delito\" o \"pena de prisión\". Asegúrate de que el resumen ejecutivo y la ficha técnica hablen de \"hacer lugar a la demanda\", "rechazar el recurso\" o \"confirmar el fallo\", reemplazando cualquier mención de \"absolución\" por términos procesales no penales adecuados.

    Borrador a auditar:
    {primer_borrador_json}

    Evidencias de respaldo del documento:
    {contexto_evidencias}

    Genera una versión final del JSON que esté perfectamente corregida, auditada y limpia.
    Responde ÚNICAMENTE con el objeto JSON final. No agregues preámbulos, explicaciones ni notas adicionales.
"""
    
    final_json_text = consultar_ollama_local(prompt_reflexion, response_json=True)
    return primer_borrador_json, final_json_text

def curar_texto_evidencia_local(texto, titulo, proposito):
    """Usa expresiones regulares en Python para unificar guiones y palabras rotas, 
    y consulta a Ollama únicamente para redactar la explicación didáctica."""
    import re
    # 1. Unimos de forma determinista y rápida palabras cortadas por saltos de línea (ej. le- galidad -> legalidad)
    texto_sanitizado = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', texto)
    
    # 2. Consultamos al modelo local únicamente para redactar la explicación didáctica
    prompt_explicacion = f"""
    Actúas como un asesor jurídico experto. Tu tarea es explicar de forma muy breve, directa y didáctica (en un máximo de dos oraciones) por qué el siguiente fragmento de un fallo judicial es relevante para la categoría '{titulo}'.
    
    Categoría legal: {titulo}
    Propósito del análisis: {proposito}
    
    Fragmento de la sentencia:
    \"{texto_sanitizado}\"
    
    Escribe tu explicación de forma clara, amigable y profesional, hablándole a un abogado. Empieza directamente con la explicación. No uses preámbulos como \"Este fragmento es relevante porque...\" ni agregues aclaraciones adicionales. Debe ser redactada 100% en español.
    """
    try:
        explicacion = consultar_ollama_local(prompt_explicacion, response_json=False)
        return texto_sanitizado, explicacion.strip()
    except Exception as e:
        print(f"Error generando explicación con Ollama: {e}")
        return texto_sanitizado, f"Este fragmento de la sentencia aporta información clave para comprender {proposito} en el marco de {titulo}."

# ==============================================================================
# SEGMENTO 6: PARSEO ROBUSTO DE JSON (EXTRACTOR ANTIFALLA CON REGEX)
# Explicación simple: A veces los modelos chicos se cansan, no cierran una comilla
# o dejan comas de más que rompen la sintaxis JSON clásica de Python. 
# Este bloque actúa como un colador de emergencia: limpia los bloques, repara comillas,
# y si todo explota, tiene expresiones regulares para pescar cada dato por separado.
# ==============================================================================

def parsear_json_robusto(texto_json):
    """Intenta parsear el JSON generado por el LLM de forma estándar. 
    Si falla, utiliza un extractor basado en expresiones regulares extremadamente tolerante 
    a errores de comas, comillas internas sin escapar, etc."""
    import re
    import json
    
    limpio = texto_json.strip()
    if limpio.startswith("```json"):
        limpio = limpio.split("```json", 1)[1]
    if limpio.endswith("```"):
        limpio = limpio.rsplit("```", 1)[0]
    limpio = limpio.strip()
    
    # Intento 1: Parseo estándar (Python)
    try:
        return json.loads(limpio)
    except Exception as e:
        print(f"DEBUG - Fallo en parseo JSON estándar: {e}. Activando extractor Regex de emergencia.")
        
    # Intento 2: Reparación básica de comillas internas desbalanceadas en los strings del JSON
    # Este regex busca valores de string e intenta escapar comillas dobles internas que no estén precedidas por backslash
    try:
        # Encontramos la sección de cada clave e intentamos sanear su contenido string
        def reparar_valor(m):
            clave = m.group(1)
            valor = m.group(2)
            # Reemplazamos comillas internas no escapadas por comillas simples
            valor_escapado = re.sub(r'(?<!\)"', "'", valor)
            return f'"{clave}": "{valor_escapado}"'
        
        reparado = re.sub(r'"(\w+)"\s*:\s*"(.*?)"(?=\s*(?:,|\}))', reparar_valor, limpio, flags=re.DOTALL)
        return json.loads(reparado)
    except Exception as e:
        print(f"DEBUG - Fallo en reparación básica: {e}. Usando extracción directa por campos.")

    # Intento 3: Extractor de emergencia campo por campo (inmune a sintaxis rota)
    datos = {
        "resumen_ejecutivo": "No se pudo formatear el resumen. Por favor reintente el análisis.",
        "juez_o_tribunal": "No especificado",
        "partes": {
            "actor_o_demandante": "No especificado",
            "demandado_o_acusado": "No especificado"
        },
        "fecha_resolucion": "No especificado",
        "decision_final": "No especificado",
        "monto_o_pena_impuesta": "No aplica"
    }
    
    # Regex lacio de captura de strings para cada clave del esquema legal solicitado
    patrones = [
        ("resumen_ejecutivo", r'"resumen_ejecutivo"\s*:\s*"(.*?)"\s*,\s*"juez_o_tribunal"'),
        ("juez_o_tribunal", r'"juez_o_tribunal"\s*:\s*"(.*?)"\s*,\s*"partes"'),
        ("actor_o_demandante", r'"actor_o_demandante"\s*:\s*"(.*?)"\s*,\s*"demandado_o_acusado"'),
        ("demandado_o_acusado", r'"demandado_o_acusado"\s*:\s*"(.*?)"\s*\}\s*,\s*"fecha_resolucion"'),
        ("fecha_resolucion", r'"fecha_resolucion"\s*:\s*"(.*?)"\s*,\s*"decision_final"'),
        ("decision_final", r'"decision_final"\s*:\s*"(.*?)"\s*,\s*"monto_o_pena_impuesta"'),
        ("monto_o_pena_impuesta", r'"monto_o_pena_impuesta"\s*:\s*"(.*?)"\s*(?:\}\s*)?$')
    ]
    
    for clave, pat in patrones:
        match = re.search(pat, limpio, re.DOTALL)
        if match:
            valor = match.group(1).strip()
            # Limpiamos comillas duplicadas
            valor = re.sub(r'^"+|"+$', '', valor)
            if clave in ["actor_o_demandante", "demandado_o_acusado"]:
                datos["partes"][clave] = valor
            else:
                datos[clave] = valor
                
    return datos

# ==============================================================================
# SEGMENTO 7: PIPELINE DE PROCESAMIENTO EN TIEMPO REAL (GENERADORES Y ESTADO)
# Explicación simple: Este bloque se encarga de conectar todos los pasos de la app
# y de avisarle a la pantalla (interfaz Gradio) en qué estado está mediante el uso
# de 'yield' (streaming). Así el usuario ve 'Paso 1/4', 'Paso 2/4', etc., mientras
# el procesador local ejecuta los cálculos.
# ==============================================================================


def pipeline_completo_local(archivo_pdf):
    if archivo_pdf is None:
        yield "Por favor, sube un archivo PDF para comenzar.", {}, "No disponible", "Sube un archivo para comenzar."
        return

    # Validamos que Ollama esté corriendo
    global ollama_client
    ollama_client = obtener_cliente_ollama()
    if ollama_client is None:
        yield f"Error: No se pudo conectar a Ollama. Asegúrate de tener Ollama Desktop abierto en segundo plano.", {}, "Error", f"Ollama no detectado en {OLLAMA_HOST_LOCAL} ni en {OLLAMA_HOST_DOCKER}."
        return

    try:
        # Paso 1: Lectura y fragmentación
        yield "Paso 1/4: Cargando y fragmentando el documento PDF página por página...", {}, "Procesando...", "Procesando..."
        almacen, cant_paginas = procesar_pdf(archivo_pdf.name)
        
        # Paso 3: Recuperación de evidencias
        yield "Paso 2/4: Buscando evidencias específicas ordenadas bajo el esquema jurídico solicitado...", {}, "Procesando...", "Procesando..."
        contexto_evidencias, evidencias_lista = recuperar_contexto_y_evidencias(almacen)
        
        # Paso 4: Generación con Ollama y Reflexión
        yield f"Paso 3/4: Consultando a Ollama ({MODELO_OLLAMA}) y ejecutando el bucle de auto-corrección reflexiva...", {}, "Procesando...", "Procesando..."
        borrador_json, final_json_text = analizar_y_reflexionar_local(contexto_evidencias)
        
        # Parseamos el JSON definitivo
        datos_finales = parsear_json_robusto(final_json_text)
        resumen = datos_finales.get("resumen_ejecutivo", "No se pudo extraer el resumen.")
        
        # Formateamos las evidencias curándolas en vivo
        texto_evidencias = "###  Fragmentos Extraídos del Fallo Original\n\n"
        texto_evidencias += f"El modelo local **{MODELO_OLLAMA}** seleccionó estas evidencias para garantizar que no haya alucinaciones.\n\n---\n\n"
        
        for i, ev in enumerate(evidencias_lista):
            yield f"Paso 4/4: Curando y estructurando de forma didáctica la Evidencia {i+1} de {len(evidencias_lista)}...", datos_finales, resumen, "Estructurando informe..."
            
            # Curación inteligente con Ollama
            texto_sanitizado, explicacion_clara = curar_texto_evidencia_local(ev['texto'], ev['titulo'], ev['proposito'])
            
            texto_evidencias += f"####  {ev['titulo']} (Documento Original: Página {ev['pagina']})\n"
            texto_evidencias += f" **¿Qué responde esta evidencia?**: {explicacion_clara}\n\n"
            texto_evidencias += f"> *\\\"...{texto_sanitizado.strip()}...\\\"*\n\n"
            texto_evidencias += "───\n\n"
            
        yield (
            f"¡Análisis completado! Modelo: {MODELO_OLLAMA}",
            datos_finales,
            resumen,
            texto_evidencias
        )
        
    except Exception as e:
        yield f"Ocurrió un error inesperado: {str(e)}", {}, "Error al procesar", f"Detalles: {str(e)}"

# ==============================================================================
# SEGMENTO 8: DISEÑO E INICIACIÓN DE LA INTERFAZ WEB (GRADIO BLOCKS)
# Explicación simple: El maquetado visual de la app. 
# Creamos dos columnas: la izquierda (Carga de archivo y botones) y la derecha
# (Fichas de resultados organizadas en pestañas de navegación intuitivas).
# Conectamos el botón con la lógica del pipeline e iniciamos el servidor web.
# ==============================================================================

with gr.Blocks(title="IA Jurídica - Ollama Local y Reflexión") as interfaz:
    gr.Markdown("# ⚖️ JusIA - Análisis de sentencias")
    gr.Markdown(f"Analizador de sentencias con OLLAMA")
    
    with gr.Row():
        # COLUMNA IZQUIERDA: Ingesta de datos
        with gr.Column(scale=1):
            input_file = gr.File(label="Subir Sentencia (PDF)", file_types=[".pdf"])
            btn_procesar = gr.Button("Analizar Documento", variant="primary")
            status_box = gr.Textbox(label="Estado del procedimiento", value="Esperando archivo...", interactive=False)
        # COLUMNA DERECHA: Resultados   
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("Resumen Ejecutivo"):
                    output_resumen = gr.Textbox(label="Resumen de Lectura Rápida", lines=5, interactive=False)
                with gr.TabItem("Ficha Técnica JSON"):
                    output_json = gr.JSON(label="Datos Estructurados del Documento")
                with gr.TabItem("Evidencias RAG"):
                    output_evidencias = gr.Markdown(value="Sube un archivo y presiona 'Analizar' para ver las evidencias ordenadas y curadas por Ollama.")

    # Evento que conecta el botón de analizar con la función del pipeline (boton para iniciar el procesamiento)                
    btn_procesar.click(
        fn=pipeline_completo_local,
        inputs=input_file,
        outputs=[status_box, output_json, output_resumen, output_evidencias]
    )

if __name__ == "__main__":
    # Arrancamos la aplicación en el puerto oficial 7860 y con el tema visual 'Soft'
    interfaz.launch(theme=gr.themes.Soft(), server_name="0.0.0.0", server_port=7860)
