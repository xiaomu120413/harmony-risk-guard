#!/bin/bash
# Build the HarmonyOS HAP (Windows, Git Bash). Requires DevEco Studio installed.
# Usage: bash harmony/scripts/build_hap.sh
set -e
DEVECO="${DEVECO_HOME:-C:/Program Files/Huawei/DevEco Studio}"
PROJ="$(cd "$(dirname "$0")/.." && pwd)"

export DEVECO_SDK_HOME="$DEVECO/sdk"
export PATH="$DEVECO/jbr/bin:$DEVECO/tools/node:$PATH"
cd "$PROJ"

NODE="node"
command -v node >/dev/null 2>&1 || NODE="$DEVECO/tools/node/node.exe"
"$NODE" "$DEVECO/tools/hvigor/bin/hvigorw.js" --no-daemon --mode module \
  -p module=entry@default -p product=default assembleHap
echo "HAP built:"
ls -l entry/build/default/outputs/default/*.hap
