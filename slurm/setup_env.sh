#!/bin/bash
# One-time environment setup for MIT Engaging/ORCD cluster.
# Run this once from the project root:
#   bash slurm/setup_env.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "=== Setting up environment for network-topology on MIT Engaging ==="

# Load miniforge for Python 3.12
module load miniforge/25.11.0-0

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv .venv
else
    echo "Virtual environment already exists."
fi

source .venv/bin/activate

echo "Installing network-topology package..."
pip install -e . --quiet

echo "Installing accelforge from PyPI..."
pip install accelforge --quiet

echo "Installing pytest for testing..."
pip install pytest --quiet

# Find ACCELFORGE_ROOT from the installed package
ACCELFORGE_PKG=$(python3 -c "import accelforge; from pathlib import Path; print(Path(accelforge.__file__).parent.parent)")
echo ""
echo "=== Setup complete ==="
echo ""
echo "AccelForge installed at: $ACCELFORGE_PKG"
echo ""
echo "To use, add to your scripts or shell:"
echo "  module load miniforge/25.11.0-0"
echo "  source $PROJECT_DIR/.venv/bin/activate"
echo "  export ACCELFORGE_ROOT=$ACCELFORGE_PKG"
echo ""
echo "Quick test:"
echo "  pytest tests/ -q"
