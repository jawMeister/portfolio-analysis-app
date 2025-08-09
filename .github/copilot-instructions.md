# Portfolio Analysis Application

Portfolio Analysis Application is a Python/Streamlit web application for analyzing and optimizing stock portfolios. It provides portfolio optimization, technical analysis, macroeconomic data integration, forecasting, and news analysis capabilities.

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Environment Setup
- **Python Version**: Requires Python 3.10-3.12. The application was developed for Python 3.10 but has compatibility issues with some dependencies in Python 3.12.
- **Critical Compatibility Issue**: `empyrical==0.5.5` does not work with Python 3.12 due to configparser API changes. Either use Python 3.10-3.11 or comment out empyrical import and Sortino ratio calculation in `src/utils.py`.

### Installation and Build Process

#### Option 1: Docker (Recommended)
```bash
# Build Docker image - takes 10-15 minutes. NEVER CANCEL. Set timeout to 30+ minutes.
./build.sh

# Run Docker container
./run-docker.sh
```

**Docker Requirements:**
- Requires internet connectivity to download base Python image and Debian packages
- If build fails with DNS resolution errors, check network connectivity
- The Dockerfile installs system dependencies: build-essential, libmagic1, poppler-utils, tesseract-ocr, pandoc

#### Option 2: Direct Python Installation
```bash
# Install dependencies - takes 5-10 minutes depending on network. NEVER CANCEL. Set timeout to 20+ minutes.
pip3 install -r requirements.txt

# Run application
./run-streamlit.sh
```

**Known Installation Issues:**
- `empyrical==0.5.5` fails on Python 3.12 - comment out in requirements.txt if needed
- Network timeouts from PyPI are common - retry installation or use Docker
- If pip install fails due to network issues, try: `pip install --timeout 60 --retries 3 -r requirements.txt`

#### Option 3: Pre-built Docker Image
```bash
# Pull from Docker Hub (if available)
docker pull jawsy/portfolio-analysis-app:latest

# Run pre-built container
docker run -p 8501:8501 jawsy/portfolio-analysis-app
```

### Running the Application
- **Streamlit URL**: http://localhost:8501
- **Development Mode**: Use `./run-streamlit.sh` which enables auto-reload on file changes
- **Production Mode**: Use Docker container for production deployment

### Configuration
- **API Keys**: Configure in `config.yaml` or `.env` file
- **Required APIs**: OpenAI, FRED, FMP, Nasdaq, Serper, Alpha Vantage
- **Environment Variables**: Set `{SERVICE}_API_KEY` environment variables
- **Configuration Check**: The app checks for API keys on startup and displays masked values

## Code Structure and Navigation

### Repository Layout
```
├── app.py                  # Main Streamlit application entry point
├── config.py              # API key management and configuration
├── config.yaml            # API key configuration file
├── requirements.txt        # Python dependencies (has merge conflict markers - clean before use)
├── build.sh               # Docker build script
├── run-streamlit.sh       # Direct Python execution script
├── run-docker.sh          # Docker run script with environment variable injection
├── Dockerfile             # Container definition
├── src/                   # Source code modules
│   ├── utils.py           # Shared utilities and data fetching
│   ├── portfolio/         # Portfolio optimization and analysis
│   ├── returns/           # Returns analysis and calculations
│   ├── macro/             # Macroeconomic data and analysis
│   ├── optimization/      # Portfolio optimization algorithms
│   ├── rebalancing/       # Portfolio rebalancing strategies
│   ├── financials/        # Financial statement analysis
│   ├── technical/         # Technical analysis indicators
│   ├── forecasting/       # Price and trend forecasting
│   └── news/              # News analysis and sentiment
├── tests/                 # Test files (minimal test coverage)
├── notebooks/             # Jupyter notebooks for experimentation
└── .gitignore             # Git ignore rules
```

### Key Files to Understand
- **`app.py`**: Main entry point, initializes Streamlit session state and renders tabs
- **`src/utils.py`**: Core utilities for data fetching (yfinance), portfolio calculations, caching
- **`config.py`**: API key management with fallback from config.yaml to environment variables
- **Module `display.py` files**: Streamlit UI for each functional area
- **Module `calculate.py` files**: Business logic and calculations

### Common Navigation Patterns
- Each functional area has a module in `src/` with `display.py`, `calculate.py`, and `plot.py`
- Streamlit session state is heavily used for data persistence across tab navigation
- Data caching using `@st.cache_data` decorator in utils.py
- Error handling with try/catch blocks and streamlit error displays

## Validation and Testing

### Manual Validation Requirements
After making changes, always test these core scenarios:

#### Basic Application Startup
1. Run `./run-streamlit.sh` or Docker container
2. Navigate to http://localhost:8501
3. Verify homepage loads with portfolio analysis interface
4. Check that all tabs render without errors: Portfolio, Returns, Macro, Optimization, Rebalancing, Financials, Technical, Forecasting, News

#### Portfolio Analysis Workflow
1. Enter stock tickers (e.g., "AAPL,GOOGL,MSFT")
2. Set date range (default: 2014-01-01 to yesterday)
3. Navigate to Portfolio tab
4. Verify portfolio optimization runs and displays:
   - Portfolio weights
   - Expected returns and risk metrics
   - Efficient frontier plot
   - Portfolio performance charts

#### Data Fetching Validation
1. Test yfinance data download for multiple tickers
2. Verify dividend data retrieval works
3. Check that caching prevents redundant API calls
4. Test error handling for invalid tickers

#### API Integration Testing (if API keys configured)
1. Test FRED API for macroeconomic data
2. Verify OpenAI integration for news analysis
3. Check other financial data APIs

### Test Infrastructure
- **Location**: `tests/` directory
- **Coverage**: Minimal test coverage exists
- **Running Tests**: No pytest configuration found - tests appear to be manually run
- **Test Files**: 
  - `tests/test_app.py` (empty)
  - `tests/src/test_utils.py`
  - `tests/src/portfolio/test_display.py`
  - `tests/src/portfolio/test_calculate.py`

### Code Validation
- **Syntax Check**: Run `python3 -m py_compile filename.py` to check syntax
- **Import Validation**: Test imports without full dependency install using: `python3 -c "import module_name"`
- **Lint Check**: No linting configuration found (flake8, black, etc.)

### Performance Considerations
- **Data Caching**: Heavy use of `@st.cache_data` for expensive operations
- **API Rate Limits**: Financial APIs have rate limits - cache aggressively
- **Memory Usage**: Large datasets from yfinance can consume significant memory
- **Computation Time**: Portfolio optimization can take 10-30 seconds for complex calculations

### Hyperparameter Tuning
- **Location**: `notebooks/` directory contains tuning scripts
- **Scripts**: 
  - `tuning_macro.sh`: Automated macroeconomic factor tuning
  - `tuning_portfolio.sh`: Portfolio optimization parameter tuning
- **Requirements**: Requires Docker image `tuning:3.11` and appropriate .env configuration
- **Usage**: These are advanced performance optimization tools for research/production

## Troubleshooting Common Issues

### Installation Problems
- **empyrical fails on Python 3.12**: Comment out import and Sortino ratio calculation in `src/utils.py` line 8 and 357
- **Network timeouts**: Use Docker approach or retry pip install with longer timeouts
- **Missing system dependencies**: Docker handles these automatically
- **requirements.txt merge conflicts**: File may contain git merge markers - clean before installation

### Runtime Issues
- **API key errors**: Check config.yaml and environment variables using `config.check_for_api_key('openai')`
- **Data download failures**: yfinance occasionally times out - retry or check network
- **Streamlit port conflicts**: Use different port: `streamlit run app.py --server.port 8502`
- **Memory issues**: Reduce date range or number of tickers for analysis
- **Cache issues**: Clear cache with `st.cache_data.clear()` in Streamlit or delete `.streamlit/` directory

### Development Workflow
- **File changes**: Use `./run-streamlit.sh` for auto-reload during development
- **Debugging**: Add logging statements using the configured logger in each module
- **Cache invalidation**: Clear Streamlit cache when data schemas change
- **Error isolation**: Test individual modules by importing them in Python REPL

### Network and Environment Issues
- **PyPI connectivity**: If pip install fails with timeout errors, this indicates firewall/network restrictions
- **Docker registry access**: If Docker build fails with DNS errors, this indicates network connectivity issues
- **Alternative approaches**: 
  - Use pre-built Docker images when available
  - Install packages individually with longer timeouts
  - Use conda instead of pip for some scientific packages
  - Consider offline installation methods for restricted environments

## Architecture Notes

### Data Flow
1. **Input**: User selects tickers and date ranges in Streamlit UI
2. **Fetching**: `src/utils.py` downloads data via yfinance with caching
3. **Processing**: Module-specific `calculate.py` files process raw data
4. **Visualization**: Module `plot.py` and `display.py` files render results
5. **Storage**: Results stored in Streamlit session state for tab navigation

### Key Dependencies
- **Streamlit 1.32.0**: Web application framework
- **yfinance 0.2.55**: Stock data provider
- **pyportfolioopt 1.5.5**: Portfolio optimization algorithms
- **pandas 2.2.1**: Data manipulation
- **plotly 5.19.0**: Interactive visualizations
- **langchain ecosystem**: AI/ML features for news and forecasting

### Security Considerations
- **API Keys**: Never commit API keys to source control
- **Environment Variables**: Use .env file for local development
- **Docker Secrets**: Use Docker secrets for production deployments

## Development Best Practices

### Code Style
- Follow existing patterns for new modules
- Use type hints where beneficial
- Add logging statements for debugging
- Handle errors gracefully with try/catch

### Streamlit Patterns
- Initialize session state in `app.py`
- Use caching decorators for expensive operations
- Test tab navigation thoroughly
- Handle widget state persistence

### Performance
- Always use `@st.cache_data` for data fetching
- Minimize API calls through caching
- Consider chunking large datasets
- Profile memory usage for large date ranges

**CRITICAL NOTES:**
- NEVER CANCEL build or installation commands - they may take 15+ minutes
- Always test complete user workflows after making changes
- Network connectivity issues are common - document workarounds
- Python 3.12 compatibility requires empyrical package fixes

## Timing Expectations

### Build and Installation Times
- **Docker build**: 10-15 minutes (up to 30 minutes with slow network) - NEVER CANCEL, set timeout to 45+ minutes
- **pip install requirements.txt**: 5-10 minutes (up to 20 minutes with network issues) - NEVER CANCEL, set timeout to 30+ minutes
- **Individual package installs**: 30 seconds to 5 minutes per package
- **Docker image pull**: 2-5 minutes depending on network speed

### Runtime Performance
- **Application startup**: 10-30 seconds depending on cache state
- **Stock data download**: 5-30 seconds for 5-10 tickers over 5-10 years
- **Portfolio optimization**: 10-30 seconds for complex optimization with 10+ assets
- **Chart rendering**: 1-5 seconds for interactive plots
- **Tab navigation**: Instant if data cached, 5-30 seconds if data needs recomputation

### Development Workflow Times
- **Code syntax check**: Immediate with `python3 -m py_compile`
- **Streamlit reload**: 2-5 seconds after file save
- **Cache clearing**: Immediate
- **Full application restart**: 10-30 seconds

## Common Outputs and References

### Repository Root Contents
```
.git/                   # Git repository data
.github/               # GitHub specific files (workflows, templates)
├── copilot-instructions.md  # This file
.gitattributes         # Git attributes configuration
.gitignore             # Git ignore patterns
Dockerfile             # Container build instructions
LICENSE                # MIT License
README.md              # Project documentation and usage
app.py                 # Main Streamlit application entry point
build.sh               # Docker build automation script
config.py              # API key and configuration management
config.yaml            # Configuration file template
docker_build_log.txt   # Build log from previous Docker builds
log.txt                # Application runtime logs
notebooks/             # Jupyter notebooks for research and tuning
requirements.txt       # Python package dependencies (150+ packages)
run-docker.sh          # Docker container execution script
run-streamlit.sh       # Direct Python execution script
src/                   # Application source code modules
tests/                 # Test files and test data
```

### Example requirements.txt Structure
The requirements.txt contains 150+ pinned dependencies including:
- Core: streamlit==1.32.0, pandas==2.2.1, numpy==1.26.4
- Financial: yfinance==0.2.55, pyportfolioopt==1.5.5, empyrical==0.5.5
- Visualization: plotly==5.19.0, matplotlib==3.8.3
- AI/ML: langchain==0.1.11, openai==1.13.3
- Analysis: scipy==1.11.4, scikit-learn==1.4.1.post1