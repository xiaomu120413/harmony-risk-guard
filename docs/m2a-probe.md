# M2-A 最小穿刺：MindSpore Lite on HarmonyOS PC — PASS

日期：2026-11-19 · 设备：HUAWEI MateBook Pro（HAD-W32）· OpenHarmony 7.1.0.25 / API 26 / arm64
工具链：DevEco Studio 26.0.1.290 · hdc 3.2.0f · MindSpore Lite converter 2.11.0（WSL2 Ubuntu 24.04）

## 1. 结论

**任务书 M2-A 通过条件（"先证明转换器与鸿蒙运行时可用"）已达成：**

- ONNX → `.ms` 转换器可用（converter_lite 2.11.0，`CONVERT RESULT SUCCESS:0`，MSL2 格式）
- 鸿蒙 PC 上 MindSpore Lite 运行时成功加载 `.ms` 并完成推理
- **数值与 PyTorch 参考完全一致**：`logits=[0.161677, -0.013502, -0.003518]`，
  max |Δ| = **4.292e-7**（float32 精度级别）

设备日志（hilog，`com.zcode.riskguard/RiskGuardProbe`）：

```
model copied: 1436 bytes -> /data/storage/el2/base/haps/entry/files/probe_mlp.ms
MODEL_LOADED inputs=1 name=input shape=[1,4]
PROBE_OK logits=[0.161677, -0.013502, -0.003518] maxdiff=4.292e-7
```

设备 UI（[截图](evidence/m2a_probe_pass.jpeg)）：`M2-A PROBE PASS` + 全部数值。

## 2. 环境事实（探测结果）

| 项 | 事实 |
|---|---|
| 设备 | HAD-W32 / HUAWEI MateBook Pro，**deviceTypes=2in1**，UDID `C551DB0E...D311` |
| 系统 | OpenHarmony-7.1.0.25，API 26，arm64-v8a |
| SDK | DevEco 26.0.1.290；`@ohos.ai.mindSporeLite` + `@kit.MindSporeLiteKit`（2in1 since API 13）|
| 设备运行时 | `/system/lib64/platformsdk/libmindspore-lite.so`、`/system/lib64/ndk/libmindspore_lite_ndk.so`、`/system/lib64/module/ai/libmindsporelite_napi.z.so` 全部存在 |
| 签名 | OpenHarmony 默认调试材料（p12/pem，密码 123456）+ 自签 profile（bundle-name 可自定义）→ **实机安装成功**（无需华为账号自动签名） |

## 3. 操作链路（全部命令可复现）

```bash
# ① 探针模型（Windows venv，固定权重 seed=42，4-16-3 MLP）
python scripts/make_probe_mlp.py
#   -> harmony/entry/src/main/resources/rawfile/probe_mlp.onnx (989B)
#   -> probe_mlp_golden.json: input [0.1,0.2,0.3,0.4] -> logits [0.161677,-0.013502,-0.003518]

# ② ONNX -> .ms（WSL，MindSpore Lite 2.11.0 Linux x64）
bash harmony/scripts/convert_onnx_to_ms.sh \
  harmony/entry/src/main/resources/rawfile/probe_mlp.onnx \
  harmony/entry/src/main/resources/rawfile/probe_mlp \
  "input:1,4"
#   -> CONVERT RESULT SUCCESS:0，probe_mlp.ms 1436B，magic "MSL2"

# ③ 构建 HAP（hvigor，Windows）
bash harmony/scripts/build_hap.sh        # BUILD SUCCESSFUL ~9s

# ④ 签名（hap-sign-tool，OpenHarmony 默认调试证书）
bash harmony/scripts/sign_hap.sh         # signApp.hap ~108KB

# ⑤ 安装 + 启动 + 日志
bash harmony/scripts/install_run.sh 3QC0124C11000711
#   install bundle successfully / start ability successfully / PROBE_OK ...
```

## 4. 关键实现点

- **rawfile → 沙箱**：MindSporeLiteKit 的 `loadModelFromFile` 只接受文件路径，
  rawfile 需先 `resourceManager.getRawFileContent('probe_mlp.ms')` 读出，
  用 `fileIo` 写入 `context.filesDir` 再加载（Index.ets）。
- **推理 API**：`loadModelFromFile` → `model.getInputs()` → `inputs[0].setData(Float32Array.buffer)`
  → `await model.predict(inputs)` → `new Float32Array(outputs[0].getData())`。
- **签名**：`hap-sign-tool.jar sign-profile`（profile 模板可自定义 bundle-name，
  device-ids 留空即可用于 OpenHarmony 调试设备）→ `sign-app`。**该设备接受
  OpenHarmony 默认调试证书，不需要华为账号自动签名。**

## 5. 产物

| 路径 | 内容 |
|---|---|
| `harmony/` | 完整可构建工程（AppScope / entry / module.json5 / ArkTS 页面 / rawfile） |
| `harmony/entry/src/main/resources/rawfile/probe_mlp.ms` | 转换产物（1436B，MSL2） |
| `harmony/tools/hapsigner/` | 签名材料 + profile 模板（bundle-name=com.zcode.riskguard） |
| `harmony/scripts/*.sh` | build / sign / install / convert 四个可复现脚本 |
| `docs/evidence/m2a_probe_pass.jpeg` | 实机 UI 截图（PROBE PASS） |

## 6. 下一步（M2-B）

1. `risk_guard.onnx`（41.5MB，[1,128] int64）→ `.ms`，核验 int64 输入、Embedding/Gather、
   LayerNorm、Attention 等算子；失败则记录最小失败算子清单并尝试 int32/静态 shape 降级。
2. M2-C：用 `tests/golden_cases.jsonl` 30 条 token IDs 做端侧 tokenizer 一致性。
3. 已知限制：本次验证的运行时加载 2.11.0 转换产物成功；若后续模型转换失败，
   需用设备匹配版本（`strings libmindspore-lite.so` 提示 2.7.0 等）重转后再测。
