#!/bin/bash
# Source this script to set up the development environment
# Usage: source activate_env.sh

# Get the absolute path of the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BUILD_DIR="$SCRIPT_DIR/build"

if [ ! -d "$BUILD_DIR" ]; then
    echo "Warning: Build directory '$BUILD_DIR' does not exist yet."
    echo "Make sure to run: mkdir build && cd build && cmake .. && make"
fi

# Add build directory to PYTHONPATH
export PYTHONPATH="$BUILD_DIR:$PYTHONPATH"

echo "Environment configured."
echo "PYTHONPATH includes: $BUILD_DIR"
