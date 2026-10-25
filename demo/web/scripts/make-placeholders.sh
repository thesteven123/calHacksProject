#!/usr/bin/env bash
# Generates synthetic placeholder media for mock mode (no real people, no dataset files).
set -euo pipefail
cd "$(dirname "$0")/../public/media"
grad() { ffmpeg -y -loglevel error -f lavfi -i "gradients=s=$1:c0=$2:c1=$3:duration=1:speed=0" -frames:v 1 -q:v 4 "$4"; }
COLS=(0x2b6cb0:0xf6ad55 0x553c9a:0x68d391 0x276749:0xfbb6ce 0x744210:0x90cdf4 0x1a365d:0xfc8181 0x285e61:0xd6bcfa)
i=0
for n in a01 a02 a03 a04 a05 a06; do
  IFS=: read -r c0 c1 <<<"${COLS[$i]}"; grad 128x128 "$c0" "$c1" "avatars/$n.jpg"; i=$((i+1))
done
i=0
for n in ig_001 ig_002 ig_003 ig_004 ig_005 ig_006; do
  IFS=: read -r c0 c1 <<<"${COLS[$i]}"; grad 640x640 "$c0" "$c1" "instagram/$n.jpg"; i=$((i+1))
done
for n in tt_001 tt_002 tt_003; do
  ffmpeg -y -loglevel error -f lavfi -i "testsrc2=s=360x640:d=6:r=25" -c:v libx264 -pix_fmt yuv420p -movflags +faststart "tiktok/$n.mp4"
done
grad 360x640 0x1a202c 0x4a5568 "tiktok/tt_004.jpg"
