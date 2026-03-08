#!/bin/bash
VENV=$(poetry env info --path)
sed -i 's/import pkg_resources//' $VENV/lib/python3.12/site-packages/webrtcvad.py
sed -i 's/__version__ = pkg_resources.get_distribution(.webrtcvad.).version/__version__ = "2.0.10"/' $VENV/lib/python3.12/site-packages/webrtcvad.py
echo "Fixed webrtcvad!"