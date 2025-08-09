# utils/common.py
import os
import hashlib

def generate_namespace(document_path: str) -> str:
    """
    Generates a consistent namespace based on the PDF file name only.
    Works for both local paths and URLs.
    """
    filename = os.path.basename(document_path)
    return hashlib.md5(filename.encode("utf-8")).hexdigest()
