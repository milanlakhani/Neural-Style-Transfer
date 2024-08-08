#!/bin/bash

# Create a virtual environment
python3 -m venv nstmil

# Activate the virtual environment
source nstmil/bin/activate

# Install the required packages
pip install -r requirements.txt
