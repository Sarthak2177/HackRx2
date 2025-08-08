#!/usr/bin/env bash
# exit on error
set -o errexit

echo "Starting build process..."

# Install Rust toolchain
echo "Installing Rust toolchain..."
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o rustup-init.sh
sh rustup-init.sh -y
# Add cargo to the path for the build process
export PATH="$HOME/.cargo/bin:$PATH"
echo "Rust toolchain installed."

# Upgrade pip
pip install --upgrade pip

# Install dependencies from requirements.txt
# Use --no-cache-dir to avoid "Read-only file system" errors
echo "Installing dependencies from requirements.txt..."
pip install --no-cache-dir -r requirements.txt
echo "Dependencies installed."

# It seems you need PyMuPDF, which is not in your requirements.txt.
# Let's install it. The original error suggests this is needed.
echo "Installing PyMuPDF..."
pip install --no-cache-dir PyMuPDF==1.22.5
echo "PyMuPDF installed."

echo "Build process completed."
