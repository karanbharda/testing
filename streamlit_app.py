#!/usr/bin/env python3
"""
Streamlit Web Interface for Indian Stock Trading Bot
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the Streamlit app function from testindia.py
from testindia import run_streamlit_app

if __name__ == "__main__":
    run_streamlit_app()
