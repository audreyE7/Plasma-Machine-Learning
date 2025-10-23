#!/usr/bin/env bash
# Usage: ./bin/ffmpeg_prep.sh input.MOV [out.mp4] [fps]
set -e
in="$1"
out="${2:-videos/plasma_fixed.mp4}"
fps="${3:-60}"
mkdir -p "$(dirname "$out")"

# If the source has HDR (Dolby Vision), tone-map to SDR BT.709 and force constant frame rate
ffmpeg -y -i "$in" \
  -vf "zscale=t=linear,tonemap=hable,zscale=p=bt709:t=bt709:m=bt709" \
  -colorspace bt709 -color_primaries bt709 -color_trc bt709 \
  -r "$fps" -vsync cfr -c:v libx264 -crf 18 -pix_fmt yuv420p \
  "$out"

echo "Wrote $out"
