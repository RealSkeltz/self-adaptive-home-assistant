#!/bin/bash
# Detect OS and set path accordingly
if [[ "$OSTYPE" == "darwin"* ]]; then
    FILE=/Users/jscheltema/Library/Caches/pypoetry/virtualenvs/home-assistant-NIV9SJTX-py3.12/lib/python3.12/site-packages/webrtcvad.py
else
    FILE=/home/realskeltz/.cache/pypoetry/virtualenvs/home-assistant-6T_GrIVA-py3.13/lib/python3.13/site-packages/webrtcvad.py
fi

echo "Fixing: $FILE"
sed -i${SED_FLAG} 's/import pkg_resources//' "$FILE"
sed -i${SED_FLAG} 's/__version__ = pkg_resources.get_distribution(.webrtcvad.).version/__version__ = "2.0.10"/' "$FILE"
echo "Done!"