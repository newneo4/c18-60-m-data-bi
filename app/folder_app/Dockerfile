#Esto configura el contenedor con Python 3.10 instalado.
FROM python:3.10-slim

#Esto copia todo lo que está en su directorio actual al directorio /app en el contenedor.
COPY . /app

#Esto establece el directorio /app como directorio de trabajo para cualquier instrucción RUN, CMD, ENTRYPOINT o COPY que siga.
WORKDIR /app

#Esto ejecuta pip install para todos los paquetes enumerados en su archivo requisitos.txt.
RUN pip install -r requirements.txt

#Esto le dice a Docker que escuche en el puerto 80 en tiempo de ejecución. El puerto 80 es el puerto estándar para HTTP.
EXPOSE 80


#Este comando le dice a Streamlit que ejecute su script app.py cuando se inicia el contenedor.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]