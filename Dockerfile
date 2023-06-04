# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

RUN apt-get update && apt-get install -y libmagic1

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

#RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
#RUN tar -xzf ta-lib-0.4.0-src.tar.gz
#WORKDIR /app/ta-lib
#RUN ./configure --prefix=/usr
#RUN make
#RUN make install

WORKDIR /app
# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# needed for langchain UnstructuredURLLoader
RUN python -m nltk.downloader -d /usr/local/share/nltk_data all

# Make port 8501 available to the world outside this container (Streamlit uses this port)
EXPOSE 8501

# Run app.py when the container launches
CMD streamlit run --server.port 8501 --server.enableCORS false app.py
