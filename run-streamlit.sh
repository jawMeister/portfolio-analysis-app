#!/bin/bash

# Run streamlit with runOnSave set so can modify files and see changes in browser
# https://docs.streamlit.io/en/stable/cli.html#cmdoption-streamlit-run-server-port
streamlit run app.py --server.port 8501 --server.runOnSave true