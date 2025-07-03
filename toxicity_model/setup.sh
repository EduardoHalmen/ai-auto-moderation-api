#!/bin/bash

# Check if virtual environment already exists
if [ -d "venv" ]; then
    :
else
    # Create virtual environment
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
# pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete. All up to date."