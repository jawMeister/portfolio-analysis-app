#!/bin/bash
docker stop portfolio-analysis
docker rm portfolio-analysis

docker run --name portfolio-analysis -p 8501:8501 portfolio-analysis-app:latest
