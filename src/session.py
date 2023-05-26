

import streamlit as st

from config import OPENAI_API_KEY, FRED_API_KEY

def check_for_openai_api_key():    
    # Initialize the API key and the flag in the session state if they are not already present
    if 'openai_api_key' not in st.session_state:
        # Import OPENAI_API_KEY only if it has a non-empty value and is not "None"
        if OPENAI_API_KEY and OPENAI_API_KEY.strip() and OPENAI_API_KEY != "None":
            st.session_state.openai_api_key = OPENAI_API_KEY
            st.session_state.openai_key_provided = True
        else:
            st.session_state.openai_api_key = ""
            st.session_state.openai_key_provided = False
            
    return (st.session_state.openai_key_provided and st.session_state.openai_api_key != "None")

def set_openai_api_key(api_key):
    st.session_state.openai_api_key = api_key
    st.session_state.openai_key_provided = True
    
    return st.session_state.openai_api_key

def get_openai_api_key():
    if check_for_openai_api_key():
        return st.session_state.openai_api_key
    else:
        return None


def check_for_fred_api_key():    
    # Initialize the API key and the flag in the session state if they are not already present
    if 'fred_api_key' not in st.session_state:
        # Import OPENAI_API_KEY only if it has a non-empty value and is not "None"
        if FRED_API_KEY and FRED_API_KEY.strip() and FRED_API_KEY != "None":
            st.session_state.fred_api_key = FRED_API_KEY
            st.session_state.fred_key_provided = True
        else:
            st.session_state.fred_api_key = ""
            st.session_state.fred_key_provided = False
            
    return (st.session_state.fred_key_provided and st.session_state.fred_api_key != "None")

def set_fred_api_key(api_key):
    st.session_state.fred_api_key = api_key
    st.session_state.fred_key_provided = True
    
    return st.session_state.fred_api_key

def get_fred_api_key(api_key):
    if check_for_fred_api_key():
        return st.session_state.fred_api_key
    else:
        return None
