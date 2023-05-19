import os
import yaml

try:
    # Try to load keys from config.yaml
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
except FileNotFoundError:
    # If the file is not found, use empty dictionary
    config = {}

# Get environment variables, or use values from config file if they're not set
OPENAI_API_KEY = os.getenv('OPEN_API_KEY', config.get('OPEN_API_KEY')) or None
FRED_API_KEY = os.getenv('FRED_API_KEY', config.get('FRED_API_KEY')) or None