#!/bin/bash
# Run the LLM Quadruped Control Frontend

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "LLM Quadruped Control - Web Frontend"
echo "=============================================="

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found"
    echo "Create one with: echo 'ANTHROPIC_API_KEY=your_key_here' > .env"
    echo ""
fi

# Install frontend dependencies if needed
echo "Checking frontend dependencies..."
pip install -q flask flask-cors python-dotenv

echo ""
echo "Starting frontend server..."
echo "Open http://localhost:5000 in your browser"
echo "Press Ctrl+C to stop"
echo "=============================================="

python frontend/app.py
