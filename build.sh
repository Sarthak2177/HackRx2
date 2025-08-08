#!/usr/bin/env bash
# Exit immediately if a command exits with a non-zero status
set -o errexit

# Set CARGO_HOME to a writable directory inside the Render build environment
export CARGO_HOME="/opt/render/project/.cargo"

# Install the Rust toolchain (no interaction, minimal output)
echo "--- Installing Rust toolchain ---"
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o rustup-init.sh
sh rustup-init.sh -y --no-modify-path --default-toolchain stable
rm rustup-init.sh

# Add Cargo binaries to PATH
export PATH="$CARGO_HOME/bin:$PATH"
echo "Rust toolchain installed and PATH configured."

# Upgrade pip
echo "--- Upgrading pip ---"
pip install --upgrade pip

# Install Python dependencies from requirements.txt
echo "--- Installing Python dependencies ---"
pip install --no-cache-dir -r requirements.txt

echo "--- Build process completed successfully! ---"
