#!/bin/bash

# Upgrade pip
pip install --upgrade pip

# Try to uninstall fitz (optional and safe)
pip uninstall -y fitz || true

# Install PyMuPDF correctly
pip install --no-cache-dir PyMuPDF==1.22.5
