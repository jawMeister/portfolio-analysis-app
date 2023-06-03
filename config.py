import os
import yaml
import streamlit as st

# use a .env file so can leverage api's in notebooks and code
from dotenv import load_dotenv
load_dotenv()

import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s (%(levelname)s):  %(module)s.%(funcName)s - %(message)s')

# Set up logger for a specific module to a different level
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

KEY_NAMES = ['openai', 'fred', 'fmp', 'nasdaq', 'serper']

def mask_key(key):
    if key: 
        if len(key) < 4:
            return '*' * len(key)
        else:
            return key[:4] + '*' * len(key[4:])
    else:
        return None

# TODO: need a way to handle authentication errors when using API keys, e.g., present the key input again

# going with a config.yaml in case we need to add other config parms, eg, for cloud deployment
# os environment variables will take priority over config.yaml
def get_key_from_config(key_name):
    key = None
    
    try:
        # Try to load keys from config.yaml
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
            masked_config = {k: mask_key(v) for k, v in config.items()}
            logger.debug(f"config: {masked_config}")
            # Ensure config is a dictionary
            assert isinstance(config, dict)
            
            key = config.get(key_name, None)
                
    except (FileNotFoundError, AssertionError):
        # If the file is not found or config is not a dictionary, use empty dictionary
        config = {}

    return key

# this whole structure allows users to bring their own API keys to the app w/o having to run the code itself
# was just having some streamlit type issues with that, need to revisit
def check_for_api_key(key_name):
    if (key_name.lower() not in KEY_NAMES):
        raise Exception(f"Invalid key_name: {key_name}, expected one of {KEY_NAMES}")
    
    env_var = f'{key_name.upper()}_API_KEY'
        
    # env var takes precedence over config.yaml
    key = os.getenv(env_var, get_key_from_config(env_var)) or None
    
    # valid key if XXX_API_KEY has a non-empty value and is not None and is not the string "None"
    api_key_provided = key and key.strip() and key != "None"

    logger.debug(f"{env_var}: {mask_key(key)} is {api_key_provided}")
    return api_key_provided
            
def set_api_key(key_name, api_key):
    if (key_name.lower() not in KEY_NAMES):
        raise Exception(f"Invalid key_name: {key_name}, expected one of {KEY_NAMES}")
    
    env_var = f'{key_name.upper()}_API_KEY'
        
    if api_key and api_key.strip() and api_key != "None":
        os.environ[env_var] = api_key
    
    logger.debug(f"{env_var}: {mask_key(os.getenv(env_var))}")
    return os.getenv(env_var)

def get_api_key(key_name):
    if (key_name.lower() not in KEY_NAMES):
        raise Exception(f"Invalid key_name: {key_name}, expected one of {KEY_NAMES}")
    
    env_var = f'{key_name.upper()}_API_KEY'
    key = os.getenv(env_var, get_key_from_config(env_var)) or None

    logger.debug(f"{env_var}: {mask_key(os.getenv(env_var))}")
    return key