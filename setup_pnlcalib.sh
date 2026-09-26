#!/bin/bash
# Fetch PnLCalib (GPL-2.0, github.com/mguti97/PnLCalib) and its single-view weights (~530 MB).
set -e
cd "$(dirname "$0")"
mkdir -p third_party
[ -d third_party/PnLCalib ] || git clone --depth 1 https://github.com/mguti97/PnLCalib third_party/PnLCalib
mkdir -p third_party/PnLCalib/weights
for w in SV_kp SV_lines; do
  [ -f third_party/PnLCalib/weights/$w ] || curl -L -o third_party/PnLCalib/weights/$w \
    https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/$w
done
echo "PnLCalib ready."
