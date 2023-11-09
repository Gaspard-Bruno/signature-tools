#!/bin/bash
if [ -e "../../../python_embeded/python" ]; then
    echo Custom Python build of ComfyUI standalone executable detected:
    echo "$(readlink -f "../../python_embeded/python")"
    echo --------------------------------------------------
    ../../../python_embeded/python pip3 install -e . --no-cache-dir
else
    echo "Custom Python not found. Use system's Python executable instead:"
    echo "$(which python3)"
    echo --------------------------------------------------
    pip3 install -e . --no-cache-dir
fi