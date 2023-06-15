#!/bin/bash
docker stop portfolio-analysis
docker rm portfolio-analysis

# Generate docker run command with all environment variables from .env
DOCKER_RUN_COMMAND="docker run --name portfolio-analysis -p 8501:8501"
if [ -f .env ]; then
    while IFS= read -r line
    do
        DOCKER_RUN_COMMAND+=" -e \"$line\""
    done < .env
fi
DOCKER_RUN_COMMAND+=" portfolio-analysis-app:latest"

echo $DOCKER_RUN_COMMAND

# Execute the generated command
eval $DOCKER_RUN_COMMAND
