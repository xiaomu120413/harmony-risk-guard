#!/bin/bash
# Install + launch the probe on a connected HarmonyOS device, then show probe logs.
# Usage: bash harmony/scripts/install_run.sh [device_id]
set -e
DEVECO="${DEVECO_HOME:-C:/Program Files/Huawei/DevEco Studio}"
HDC="$DEVECO/sdk/default/openharmony/toolchains/hdc.exe"
PROJ="$(cd "$(dirname "$0")/.." && pwd)"
BUNDLE="com.zcode.riskguard"

DEV="${1:-}"
if [ -z "$DEV" ]; then
  DEV=$("$HDC" list targets | head -1)
fi
echo "device: $DEV"

MSYS_NO_PATHCONV=1 "$HDC" -t "$DEV" install -r "$PROJ/tools/hapsigner/signApp.hap"
"$HDC" -t "$DEV" shell "aa start -a EntryAbility -b $BUNDLE"
sleep 3
"$HDC" -t "$DEV" shell "hilog -x | grep -E 'RiskGuardProbe|PROBE_OK|PROBE_FAIL|MODEL_LOADED' | tail -10"
