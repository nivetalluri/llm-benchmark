#!/bin/bash
# LLM Bench — one-shot dependency setup (run inside your venv if you use one)
set -e
echo "Installing LLM Bench dependencies..."
pip install fastapi "uvicorn[standard]" psutil requests openpyxl \
            python-multipart pypdf python-docx pillow nvidia-ml-py
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
echo ""
echo "Done. Start the app with:"
echo "  python3 -m uvicorn app:app --host 0.0.0.0 --port 8000"
echo ""
echo "Optional (image OCR):  sudo apt install tesseract-ocr && pip install pytesseract"
echo "Optional (GPU/CUDA):   reinstall llama-cpp-python with CMAKE_ARGS=\"-DGGML_CUDA=on\""
