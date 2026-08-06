# 1. Usamos la imagen oficial ligera de Python 3.12
FROM python:3.12-slim

# 2. Definimos el directorio de trabajo dentro del contenedor
WORKDIR /app

# 3. Copiamos el archivo de dependencias primero para optimizar la caché de Docker
COPY requirements.txt .

# 4. Instalamos las dependencias necesarias de Python
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiamos el archivo de la aplicación principal y los scripts auxiliares
COPY app.py .

# 6. Exponemos el puerto oficial en el que corre la interfaz web de Gradio
EXPOSE 7860

# 7. Ejecutamos la aplicación
CMD ["python", "app.py"]