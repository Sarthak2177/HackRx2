#!/usr/bin/env bash
# exit on error
set -o errexit

# Set CARGO_HOME to a writable directory. This is the crucial fix.
export CARGO_HOME="/opt/render/project/.cargo"

# Install the Rust toolchain
echo "--- Installing Rust toolchain ---"
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o rustup-init.sh
sh rustup-init.sh -y --no-modify-path
# Add the new cargo bin directory to the PATH
export PATH="$CARGO_HOME/bin:$PATH"
echo "Rust toolchain configured."

# Upgrade pip
echo "--- Upgrading pip ---"
pip install --upgrade pip

# Install Python dependencies from requirements.txt
# The --no-cache-dir flag is essential for pip on Render
echo "--- Installing Python dependencies ---"
pip install --no-cache-dir -r requirements.txt

echo "--- Build process completed successfully! ---"
