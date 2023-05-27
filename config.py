import os
import yaml
import streamlit as st

import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s (%(levelname)s):  %(module)s.%(funcName)s - %(message)s')

# Set up logger for a specific module to a different level
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def mask_key(key):
    return key[:4] + '*' * len(key[4:])

try:
    # Try to load keys from config.yaml
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        masked_config = {k: mask_key(v) for k, v in config.items()}
        logger.debug(f"config: {masked_config}")
        # Ensure config is a dictionary
        assert isinstance(config, dict)
            
except (FileNotFoundError, AssertionError):
    # If the file is not found or config is not a dictionary, use empty dictionary
    config = {}

# Get environment variables, or use values from config file if they're not set
FRED_API_KEY = os.getenv('FRED_API_KEY', config.get('FRED_API_KEY')) or None
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', config.get('OPENAI_API_KEY')) or None
FMP_API_KEY = os.getenv('FMP_API_KEY', config.get('FMP_API_KEY')) or None

logger.debug(f"FRED_API_KEY: {mask_key(FRED_API_KEY) if FRED_API_KEY else None}")
logger.debug(f"OPENAI_API_KEY: {mask_key(OPENAI_API_KEY) if OPENAI_API_KEY else None}")
logger.debug(f"FMP_API_KEY: {mask_key(FMP_API_KEY) if FMP_API_KEY else None}")