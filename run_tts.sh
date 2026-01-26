#!/bin/bash
# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment
source venv/bin/activate

# Set environment variables for memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the Gradio app
python3 app.py
