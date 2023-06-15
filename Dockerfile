# Use an official Python runtime as a parent image
FROM python:3.11.4-slim-buster

# believe is needed by UnstructuredURLLoader - from https://unstructured-io.github.io/unstructured/installing.html
# UnstructuredURLLoader can still hang on some URLs even with these installed
RUN apt-get update && apt-get install -y build-essential \  
                                            libmagic1 \
                                            libmagic-dev \
                                            poppler-utils \
                                            tesseract-ocr \
                                            libreoffice \
                                            pandoc \
                                            libxml2 \
                                            libxslt1-dev

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD app.py /app
ADD config.py /app
ADD requirements.txt /app
ADD src /app/src

WORKDIR /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# needed for langchain UnstructuredURLLoader
RUN python -m nltk.downloader -d /usr/local/share/nltk_data all

# Make port 8501 available to the world outside this container (Streamlit uses this port)
EXPOSE 8501

# Run app.py when the container launches
ENTRYPOINT ["streamlit", "run", "--server.port", "8501", "--server.enableCORS", "true", "app.py"]
