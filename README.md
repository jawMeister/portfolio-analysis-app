# Portfolio Analysis Application

This is a portfolio analysis application that calculates optimal portfolios for a given set of tickers and time periods. It uses historical stock data to generate portfolios and provides various visualizations to analyze the results. Currently iterating towards a target state application.

## Usage

To use this application, you can build from source or need to have Docker installed on your machine.

1. To pull the Docker image:

   ```shell
   docker pull jawsy/portfolio-analysis-app:latest

   Alternatively:
      build from source using pip install -r requirements.txt
      or build a docker container with build.sh

2. Run the Docker container:

   ```shell
   docker run -p 8501:8501 jawsy/portfolio-analysis-app

   Alternatively, 
      if built from source, run-streamlit.sh
      if built a container, run-docker.sh

3. Open your web browser and visit http://localhost:8501 to access the application.
4. On the application's homepage, you will see various visualizations and analysis options.
5. Select the desired time period and tickers for the portfolio analysis.
6. Click on the different tabs and visualizations to explore the results.
7. Experiment with different time periods, tickers, and analysis options to gain insights into portfolio performance.

## Acknowledgments

- The portfolio analysis algorithms and visualizations were developed by jawsy.
- The historical stock data is obtained from yfinance.
- This application is for educational and informational purposes only. It does not constitute financial advice.

## License

This project is licensed under the MIT License.