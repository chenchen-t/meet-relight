#!/usr/bin/env bash
# Builds the Tensor API graph generator inside the LiteRT workspace and writes
# the .tflite graphs into apps/demo/public/models.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LITERT="$ROOT/third_party/LiteRT"
OUT="$ROOT/apps/demo/public/models"
[ -d "$LITERT" ] || git clone --depth 1 https://github.com/google-ai-edge/LiteRT.git "$LITERT"
ln -sfn "$ROOT/native/tensor_graphs" "$LITERT/tensor/meet_relight"
mkdir -p "$OUT"
cd "$LITERT"
bazelisk run -c opt //tensor/meet_relight:meet_relight_graphs -- --out_dir="$OUT" --h="${H:-288}" --w="${W:-512}"
ls -la "$OUT"/seg_*.tflite

# MediaPipe face landmark model ships with a dynamic (-1) batch; pin it for LiteRT.js GPU-buffer binding.
FL="$OUT/face_landmarker"
if [ -f "$FL/face_landmarks_detector.tflite" ]; then
  bazelisk run -c opt //tensor/meet_relight:static_batch -- "$FL/face_landmarks_detector.tflite" "$FL/face_landmarks_detector_b1.tflite"
fi
