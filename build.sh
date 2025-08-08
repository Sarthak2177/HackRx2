bash
Copy
Edit
#!/bin/bash
pip install --upgrade pip
pip uninstall -y fitz || true
pip install --no-cache-dir PyMuPDF==1.22.5
