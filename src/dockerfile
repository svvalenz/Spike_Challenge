# Usa una imagen base. Por ejemplo, una imagen de Python.
FROM python:3.12
# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app
# Copia la carpeta entera desde el contexto de construcción al directorio de trabajo
COPY ./src /app/src
COPY requirements.txt /app
# Instala las dependencias necesarias
RUN pip install -r /app/requirements.txt
# Comando para ejecutar la aplicación
CMD ["python", "/app/src/modelo.py"]