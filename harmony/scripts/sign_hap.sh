#!/bin/bash
# Sign the unsigned HAP with OpenHarmony default debug materials (password 123456).
# Works on OpenHarmony debug devices (verified on HUAWEI MateBook Pro / HAD-W32, OH 7.1 / API 26).
# Usage: bash harmony/scripts/sign_hap.sh
set -e
DEVECO="${DEVECO_HOME:-C:/Program Files/Huawei/DevEco Studio}"
PROJ="$(cd "$(dirname "$0")/.." && pwd)"
MAT="$PROJ/tools/hapsigner"
JAVA="$DEVECO/jbr/bin/java"
TOOL="$DEVECO/sdk/default/openharmony/toolchains/lib/hap-sign-tool.jar"
IN="$PROJ/entry/build/default/outputs/default/entry-default-unsigned.hap"
OUT="$MAT/signApp.hap"

"$JAVA" -jar "$TOOL" sign-profile \
  -mode localSign \
  -keyAlias "OpenHarmony Application Profile Debug" \
  -keyPwd "123456" \
  -inFile "$MAT/UnsgnedDebugProfileTemplate.json" \
  -outFile "$MAT/ohos_provision_debug.p7b" \
  -keystoreFile "$MAT/OpenHarmony.p12" \
  -keystorePwd "123456" \
  -signAlg "SHA256withECDSA" \
  -profileCertFile "$MAT/OpenHarmonyProfileDebug.pem"

"$JAVA" -jar "$TOOL" sign-app \
  -keyAlias "openharmony application release" \
  -signAlg "SHA256withECDSA" \
  -mode localSign \
  -appCertFile "$MAT/OpenHarmonyApplication.pem" \
  -profileFile "$MAT/ohos_provision_debug.p7b" \
  -inFile "$IN" \
  -keystoreFile "$MAT/OpenHarmony.p12" \
  -outFile "$OUT" \
  -keyPwd "123456" \
  -keystorePwd "123456"
echo "Signed HAP: $OUT"
ls -l "$OUT"
