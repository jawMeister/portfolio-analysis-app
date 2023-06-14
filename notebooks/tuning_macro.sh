#!/bin/bash

# Factors array
declare -a FACTORS=("Federal Funds Rate" "Unemployment Rate" "CPI" "PCE" "Retail Sales" "Initial Claims" "Housing Starts" "5-Year Forward Inflation Expectation Rate" "Economic Policy Uncertainty Index for United States" "10-Year Treasury Constant Maturity Rate" "GDP")

# Initial window in days array
declare -a INITIAL_WINDOWS=(3652 2922 2192 1826 1462)

# Iterate over factors
for INITIAL_WINDOW_IN_DAYS in "${INITIAL_WINDOWS[@]}"; do
  counter=0
  # Iterate over initial windows
  for FACTOR in "${FACTORS[@]}"; do
    echo "Running docker for factor: $FACTOR and initial window: $INITIAL_WINDOW_IN_DAYS"
    counter=$((counter+1))

    # Replace spaces with underscores in factor for container name
    container_name="tuning-${INITIAL_WINDOW_IN_DAYS}-${counter}"
    echo "Container name: ($container_name)"

    # If container exists, stop and remove it
    if [ "$(docker ps -aq -f name=$container_name)" ]; then
      docker stop $container_name
      docker rm $container_name
    fi

    # Update FACTOR in .env file
    sed -i "s/^FACTOR=.*/FACTOR=\"$FACTOR\"/" .env

    # Source .env file
    source .env
    echo "START_DATE: $START_DATE, END_DATE: $END_DATE, for FACTOR: $FACTOR, INITIAL_WINDOW_IN_DAYS: $INITIAL_WINDOW_IN_DAYS, TICKERS: $TICKERS, FRED_API_KEY: $FRED_API_KEY, NASDAQ_API_KEY: $NASDAQ_API_KEY"

    tmp_dir=./tmp-tune/$counter/$INITIAL_WINDOW_IN_DAYS
    mkdir -p $tmp_dir

    cpus_to_use=$(($(nproc --all) * 8 / 10))
    echo "Using $cpus_to_use CPUs"

    docker run --name $container_name --cpus=$cpus_to_use -e INITIAL_WINDOW_IN_DAYS=$INITIAL_WINDOW_IN_DAYS -e START_DATE=$START_DATE -e END_DATE=$END_DATE -e FRED_API_KEY=$FRED_API_KEY -e NASDAQ_API_KEY=$NASDAQ_API_KEY -v $(pwd)/.env:/app/.env -e OUTPUT_PATH="/app/output/hyperparms" -v $tmp_dir:/tmp -v ./hyperparm-files:/app/output/hyperparms tuning:3.11
    echo "Executed docker run for factor: $FACTOR and initial window: $INITIAL_WINDOW_IN_DAYS"

    # Wait for the docker container to finish
    docker wait $container_name
    echo "Docker container finished for factor: $FACTOR and initial window: $INITIAL_WINDOW_IN_DAYS"

    rm -rf $tmp_dir

    # Check if container exists and if so, remove it
    if [ "$(docker ps -aq -f status=exited -f name=$container_name)" ]; then
      docker rm $container_name
      echo "Removed docker container for factor: $FACTOR and initial window: $INITIAL_WINDOW_IN_DAYS"
    fi
  done
done



