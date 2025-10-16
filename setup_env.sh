#!/bin/bash

# Environment Setup Script for Diversity Distillation Project
# This script sets up both local development and Azure ML compatible environments

set -e  # Exit on any error

echo "🚀 Setting up Diversity Distillation Environment..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not found. Please install Python 3.8 or higher."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements based on environment
if [ "$1" = "azure" ]; then
    echo "☁️  Installing Azure ML requirements..."
    pip install -r requirements-azure.txt
else
    echo "💻 Installing local development requirements..."
    pip install -r requirements-local.txt
fi

# Verify key packages are installed
echo "🔍 Verifying installation..."
python3 -c "
import torch
import diffusers
import transformers
import matplotlib
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ Diffusers: {diffusers.__version__}')
print(f'✅ Transformers: {transformers.__version__}')
print('✅ All key packages installed successfully')
"

echo "🎉 Environment setup complete!"
echo ""
echo "To activate the environment, run:"
echo "source .venv/bin/activate"
echo ""
echo "To run with Azure ML packages:"
echo "./setup_env.sh azure"