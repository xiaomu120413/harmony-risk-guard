#!/bin/bash
# ONNX -> MindSpore Lite .ms (run inside WSL/Linux with the converter extracted).
# MindSpore Lite 2.11.0 Linux x64:
#   https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.11.0/MindSporeLite/lite/release/linux/x86_64/mindspore-lite-2.11.0-linux-x64.tar.gz
# Usage: bash harmony/scripts/convert_onnx_to_ms.sh <input.onnx> <output_prefix> [input_shape]
set -e
MSL="${MSL_ROOT:-$HOME/mindspore-lite-2.11.0-linux-x64}"
CONV="$MSL/tools/converter/converter/converter_lite"
IN="${1:?input .onnx}"
OUT="${2:?output prefix (no extension)}"
SHAPE="${3:-}"
export LD_LIBRARY_PATH="$MSL/tools/converter/lib:$MSL/runtime/lib:$LD_LIBRARY_PATH"
ARGS=(--fmk=ONNX --modelFile="$IN" --outputFile="$OUT")
if [ -n "$SHAPE" ]; then
  ARGS+=(--inputShape="$SHAPE")
fi
"$CONV" "${ARGS[@]}"
ls -l "${OUT}.ms"
