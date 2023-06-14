#!/bin/bash

# Initial window in days array
declare -a initial_windows=(3652 2922 2192 1826 1462)

counter=0
# Iterate over factors
for INITIAL_WINDOW_IN_DAYS in "${initial_windows[@]}"; do
    FACTOR="Portfolio"

    # Update FACTOR variable in .env file
    sed -i "s/^FACTOR=.*/FACTOR=$FACTOR/" .env

    # Read the .env file to get the rest of the variables (eg, API KEYS)
    source .env

    # Iterate over initial windows
    echo "Running docker for tickers: $TICKERS and initial window: $INITIAL_WINDOW_IN_DAYS"
    counter=$((counter+1))

    # Replace spaces with underscores in factor for container name
    container_name="tuning-${TICKERS}-${INITIAL_WINDOW_IN_DAYS}-${counter}"
    echo "Container name: ($container_name)"

    # If container exists, stop and remove it
    if [ "$(docker ps -aq -f name=$container_name)" ]; then
      docker stop $container_name
      docker rm $container_name
    fi

    tmp_dir=./tmp-tune/portfolio/$counter/$initial_window
    mkdir -p $tmp_dir

    cpus_to_use=$(($(nproc --all) * 8 / 10))
    echo "Using $cpus_to_use CPUs"

    docker run --name $container_name --cpus=$cpus_to_use -e TICKERS=$TICKERS -e INITIAL_WINDOW_IN_DAYS=$INITIAL_WINDOW_IN_DAYS -e START_DATE=$START_DATE -e END_DATE=$END_DATE -e FRED_API_KEY=$FRED_API_KEY -e NASDAQ_API_KEY=$NASDAQ_API_KEY -v $(pwd)/.env:/app/.env -e OUTPUT_PATH="/app/output/hyperparms" -v $tmp_dir:/tmp -v ./hyperparm-files:/app/output/hyperparms tuning:3.11
    echo "Executed docker run for factor: $FACTOR and initial window: $INITIAL_WINDOW_IN_DAYS"

    # Wait for the docker container to finish
    docker wait $container_name
    echo "Docker container finished for factor: $FACTOR and initial_window: $INITIAL_WINDOW_IN_DAYS"

    rm -rf $tmp_dir

    # Check if container exists and if so, remove it
    if [ "$(docker ps -aq -f status=exited -f name=$container_name)" ]; then
      docker rm $container_name
      echo "Removed docker container for factor: $FACTOR and initial_window: $INITIAL_WINDOW_IN_DAYS"
    fi

done


