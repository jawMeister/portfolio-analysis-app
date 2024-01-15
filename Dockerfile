# Use an official Python runtime as a parent image
FROM python:3-slim-bullseye

# believe is needed by UnstructuredURLLoader - from https://unstructured-io.github.io/unstructured/installing.html
# UnstructuredURLLoader can still hang on some URLs even with these installed
#RUN apt-get update && apt-get install -y build-essential \  
#                                            libmagic1 \
#                                            libmagic-dev \
#                                            poppler-utils \
#                                            tesseract-ocr \
#                                            libreoffice \
#                                            pandoc \
#                                            libxml2 \
#                                            libxslt1-dev

RUN apt-get update
RUN echo "installing build-essential"
RUN apt-get install -y build-essential
RUN echo "installing libmagic1"
RUN apt-get install -y libmagic1
RUN echo "installing libmagic-dev"
RUN apt-get install -y libmagic-dev
RUN echo "installing poppler-utils"
RUN apt-get install -y poppler-utils
RUN echo "installing tesseract-ocr"
RUN apt-get install -y tesseract-ocr
# 01/06/23: libreoffice not installing correctly, commenting out for now
#RUN echo "installing libreoffice"
#RUN apt-get install -y libreoffice
RUN echo "installing pandoc"
RUN apt-get install -y pandoc
RUN echo "installing libxml2"
RUN apt-get install -y libxml2
RUN echo "installing libxslt1-dev"
RUN apt-get install -y libxslt1-dev

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD app.py /app
ADD config.py /app
ADD requirements.txt /app
ADD src /app/src

WORKDIR /app

RUN pip3 install --upgrade setuptools
RUN pip3 install --upgrade pip

# Install any needed packages specified in requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# needed for langchain UnstructuredURLLoader
RUN python3 -m nltk.downloader -d /usr/local/share/nltk_data all

# Make port 8501 available to the world outside this container (Streamlit uses this port)
EXPOSE 8501

# Run app.py when the container launches
ENTRYPOINT ["streamlit", "run", "--server.port", "8501", "--server.enableCORS", "true", "app.py"]
