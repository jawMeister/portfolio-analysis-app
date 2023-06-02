

import streamlit as st

import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s (%(levelname)s):  %(module)s.%(funcName)s - %(message)s')

# Set up logger for a specific module to a different level
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from config import OPENAI_API_KEY, FRED_API_KEY, FMP_API_KEY, mask_key

def check_for_openai_api_key():    
    # Initialize the API key and the flag in the session state if they are not already present
    if 'openai_api_key' not in st.session_state:
        logger.debug(f"Initializing OPENAI_API_KEY: {OPENAI_API_KEY}")
        # Import OPENAI_API_KEY only if it has a non-empty value and is not "None"
        if OPENAI_API_KEY and OPENAI_API_KEY.strip() and OPENAI_API_KEY != "None":
            st.session_state.openai_api_key = OPENAI_API_KEY
            st.session_state.openai_key_provided = True
        else:
            st.session_state.openai_api_key = ""
            st.session_state.openai_key_provided = False
            
    logger.debug(f"OPENAI_API_KEY: {mask_key(st.session_state.openai_api_key)}, openai_key_provided: {st.session_state.openai_key_provided}")
    return (st.session_state.openai_key_provided and st.session_state.openai_api_key != "None" and st.session_state.openai_api_key != None)

def set_openai_api_key(api_key):
    if api_key and api_key.strip() and api_key != "None":
        st.session_state.openai_api_key = api_key
        st.session_state.openai_key_provided = True
    
    logger.debug(f"OPENAI_API_KEY: {mask_key(st.session_state.openai_api_key)}, openai_key_provided: {st.session_state.openai_key_provided}")
    return st.session_state.openai_api_key

def get_openai_api_key():
    key = None
    
    if check_for_openai_api_key():
        return st.session_state.openai_api_key

    logger.debug(f"OPENAI_API_KEY: {mask_key(st.session_state.openai_api_key)}, openai_key_provided: {st.session_state.openai_key_provided}")
    return key
    
def openai_api_key_provided():
    logger.debug(f"OPENAI_API_KEY: {mask_key(st.session_state.openai_api_key)}, openai_key_provided: {st.session_state.openai_key_provided}")
    return st.session_state.openai_key_provided

def check_for_fred_api_key():    
    # Initialize the API key and the flag in the session state if they are not already present
    if 'fred_api_key' not in st.session_state:
        logger.debug(f"Initializing FRED_API_KEY: {FRED_API_KEY}")
        # Import OPENAI_API_KEY only if it has a non-empty value and is not "None"
        if FRED_API_KEY and FRED_API_KEY.strip() and FRED_API_KEY != "None":
            st.session_state.fred_api_key = FRED_API_KEY
            st.session_state.fred_key_provided = True
        else:
            st.session_state.fred_api_key = ""
            st.session_state.fred_key_provided = False
    
    logger.debug(f"FRED_API_KEY: {mask_key(st.session_state.fred_api_key)}, fred_key_provided: {st.session_state.fred_key_provided}")
    return (st.session_state.fred_key_provided and st.session_state.fred_api_key != "None" and st.session_state.fred_api_key != None)

def set_fred_api_key(api_key):
    if api_key and api_key.strip() and api_key != "None":
        st.session_state.fred_api_key = api_key
        st.session_state.fred_key_provided = True
    
    logger.debug(f"FRED_API_KEY: {mask_key(st.session_state.fred_api_key)}, fred_key_provided: {st.session_state.fred_key_provided}")
    return st.session_state.fred_api_key

def get_fred_api_key():
    key = None
    
    if check_for_fred_api_key():
        key = st.session_state.fred_api_key
    
    logger.debug(f"FRED_API_KEY: {mask_key(st.session_state.fred_api_key)}, fred_key_provided: {st.session_state.fred_key_provided}")
    return key

def fred_api_key_provided():
    logger.debug(f"FRED_API_KEY: {mask_key(st.session_state.fred_api_key)}, fred_key_provided: {st.session_state.fred_key_provided}")
    return st.session_state.fred_key_provided

def check_for_fmp_api_key():    
    # Initialize the API key and the flag in the session state if they are not already present
    if 'fmp_api_key' not in st.session_state:
        logger.debug(f"Initializing FMP_API_KEY: {FMP_API_KEY}")
        # Import OPENAI_API_KEY only if it has a non-empty value and is not "None"
        if FMP_API_KEY and FMP_API_KEY.strip() and FMP_API_KEY != "None":
            st.session_state.fmp_api_key = FMP_API_KEY
            st.session_state.fmp_api_key_provided = True
        else:
            st.session_state.fmp_api_key = ""
            st.session_state.fmp_api_key_provided = False
    
    logger.debug(f"FMP_API_KEY: {mask_key(st.session_state.fmp_api_key)}, fmp_key_provided: {st.session_state.fmp_api_key_provided}")
    return (st.session_state.fmp_api_key_provided and st.session_state.fmp_api_key != "None" and st.session_state.fmp_api_key != None)

def set_fmp_api_key(api_key):
    if api_key and api_key.strip() and api_key != "None":
        st.session_state.fmp_api_key = api_key
        st.session_state.fmp_api_key_provided = True
    
    logger.debug(f"FMP_API_KEY: {mask_key(st.session_state.fmp_api_key)}, fmp_key_provided: {st.session_state.fmp_api_key_provided}")
    return st.session_state.fmp_api_key

def get_fmp_api_key():
    key = None
    
    if check_for_fmp_api_key():
        key = st.session_state.fmp_api_key
    
    logger.debug(f"FMP_API_KEY: {mask_key(st.session_state.fmp_api_key)}, fmp_key_provided: {st.session_state.fmp_api_key_provided}")
    return key

def fmp_api_key_provided():
    logger.debug(f"FMP_API_KEY: {mask_key(st.session_state.fmp_api_key)}, fmp_key_provided: {st.session_state.fmp_api_key_provided}")
    return st.session_state.fmp_api_key_provided