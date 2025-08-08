#!/bin/bash
pip install --upgrade pip
pip uninstall -y fitz || true
pip install -r requirements.txt
