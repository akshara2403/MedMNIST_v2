#!/usr/bin/env bash
# Get the authors' code (pinned commit), install its dependencies, apply the small documented patch.
set -euo pipefail
cd "$(dirname "$0")"
COMMIT=70b6b3a7ad7afddff1df2a3b735235830fbdb142
[ -d official_repo ] || git clone -q https://github.com/MedMNIST/experiments.git official_repo
git -C official_repo checkout -q "$COMMIT"
pip install -q tensorboardX "git+https://github.com/M3DV/ACSConv.git"
python patch_official.py official_repo
echo "Official code ready in official_repo/ (diff: official_patch.diff)"
