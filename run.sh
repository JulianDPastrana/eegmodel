#!/bin/bash
# Quick start script for the EEG model training project

echo "=========================================="
echo "EEG Model Training - Quick Start"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found!"
    echo "Please run: python3.11 -m venv venv"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check Python version
python_version=$(python --version)
echo "✓ $python_version"

# Display options
echo ""
echo "What would you like to do?"
echo ""
echo "1) Run tests"
echo "2) Train EEGConformer on BCI IV 2a (Subject 1)"
echo "3) Train EEGConformer on ADHD dataset"
echo "4) Run getting started example"
echo "5) Install/update dependencies"
echo "6) Exit"
echo ""
read -p "Enter your choice (1-6): " choice

case $choice in
    1)
        echo ""
        echo "Running tests..."
        python tests/test_models.py
        python tests/test_training.py
        ;;
    2)
        echo ""
        echo "Training EEGConformer on BCI IV 2a dataset..."
        python experiments/train_conformer.py --dataset bci --subject 1 --epochs 20
        ;;
    3)
        echo ""
        echo "Training EEGConformer on ADHD dataset..."
        python experiments/train_conformer.py --dataset adhd --epochs 20
        ;;
    4)
        echo ""
        echo "Running getting started example..."
        python notebooks/getting_started.py
        ;;
    5)
        echo ""
        echo "Installing/updating dependencies..."
        pip install --upgrade pip
        pip install -r requirements.txt
        ;;
    6)
        echo "Goodbye!"
        exit 0
        ;;
    *)
        echo "Invalid choice!"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
