#!/bin/bash
# Sign the unsigned HAP with an OpenHarmony *release* profile (normal_hap domain).
# Unlike the debug profile, this gives the app the normal SELinux domain, which is
# required for NPU/NNRT HDI access on HarmonyOS PC.
# Usage: bash harmony/scripts/sign_hap_release.sh
set -e
DEVECO="${DEVECO_HOME:-C:/Program Files/Huawei/DevEco Studio}"
PROJ="$(cd "$(dirname "$0")/.." && pwd)"
MAT="$PROJ/tools/hapsigner"
JAVA="$DEVECO/jbr/bin/java"
TOOL="$DEVECO/sdk/default/openharmony/toolchains/lib/hap-sign-tool.jar"
IN="$PROJ/entry/build/default/outputs/default/entry-default-unsigned.hap"
OUT="$MAT/signAppRelease.hap"

"$JAVA" -jar "$TOOL" sign-profile \
  -mode localSign \
  -keyAlias "OpenHarmony Application Profile Release" \
  -keyPwd "123456" \
  -inFile "$MAT/UnsgnedReleasedProfileTemplate.json" \
  -outFile "$MAT/ohos_provision_release.p7b" \
  -keystoreFile "$MAT/OpenHarmony.p12" \
  -keystorePwd "123456" \
  -signAlg "SHA256withECDSA" \
  -profileCertFile "$MAT/OpenHarmonyProfileRelease.pem"

"$JAVA" -jar "$TOOL" sign-app \
  -keyAlias "openharmony application release" \
  -signAlg "SHA256withECDSA" \
  -mode localSign \
  -appCertFile "$MAT/OpenHarmonyApplication.pem" \
  -profileFile "$MAT/ohos_provision_release.p7b" \
  -inFile "$IN" \
  -keystoreFile "$MAT/OpenHarmony.p12" \
  -outFile "$OUT" \
  -keyPwd "123456" \
  -keystorePwd "123456"
echo "Release-signed HAP: $OUT"
ls -l "$OUT"
