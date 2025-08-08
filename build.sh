#!/usr/bin/env bash
set -o errexit

# Writable Cargo home
export CARGO_HOME="/opt/render/project/.cargo"

# Install Rust
echo "--- Installing Rust toolchain ---"
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o rustup-init.sh
sh rustup-init.sh -y --no-modify-path --default-toolchain stable
rm rustup-init.sh

# Add Cargo to PATH
export PATH="$CARGO_HOME/bin:$PATH"

# Confirm Cargo works
cargo --version
rustc --version

# Upgrade pip
echo "--- Upgrading pip ---"
pip install --upgrade pip

# Install maturin first to avoid pip calling it without CARGO_HOME
pip install --no-cache-dir maturin

# Install project dependencies
echo "--- Installing Python dependencies ---"
pip install --no-cache-dir -r requirements.txt

echo "--- Build completed successfully ---"
