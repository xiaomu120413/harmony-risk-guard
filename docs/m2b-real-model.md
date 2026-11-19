# M2-B：真实模型 MiniRBT 端侧推理 — PASS

日期：2026-11-19 · 设备：HUAWEI MateBook Pro（HAD-W32）/ OpenHarmony 7.1.0.25 / API 26 / arm64
模型：`models/risk_guard.onnx`（MiniRBT-h256 风险分类器，41.5MB）→ `risk_guard.ms`（41.2MB，MSL2）

## 1. 结论

**任务书 M2-B（MiniRBT ONNX 算子/类型核验）与 M2-D/E 的核心推理路径已达成：**

- ONNX 模型（int64 [1,128] × 2 输入）经 MindSpore Lite converter 2.11.0 **一次转换成功**，
  无算子报错
- 实机加载 + 推理成功：**load 24ms / infer 17ms**
- **数值与 PyTorch 参考一致**：logits `[-1.752675, -3.063437, 4.612434]`
  vs PyTorch `[-1.752671, -3.063433, 4.612427]`，max |Δ| = **7.387e-6**
- 标签一致：BLOCK（与 PyTorch 相同）
- 关键发现：**converter 自动将 int64 输入降为 int32**（设备上报 dtype=34/4B）；
  端侧填充逻辑按 dtype 自适应（int64→8 字节、int32→4 字节），无需改模型

设备日志：

```
MLP_OK logits=[0.161677, -0.013502, -0.003518] maxdiff=4.292e-7
RG_OK sample=HRG-DATA-04-00 logits=[-1.752675, -3.063437, 4.612434]
      label=BLOCK (pt=BLOCK) maxdiff=7.387e-6 load=24ms infer=17ms
      inputs=2 dtype0=34/4B dtype1=34/4B
```

UI（[截图](evidence/m2b_real_model_pass.jpeg)）：同一页面 M2-A/M2-B 双双 PASS。

## 2. 执行细节

```bash
# ① 转换（WSL，converter_lite 2.11.0）
bash harmony/scripts/convert_onnx_to_ms.sh models/risk_guard.onnx \
  harmony/entry/src/main/resources/rawfile/risk_guard "input_ids:1,128"
#   -> CONVERT RESULT SUCCESS:0, 41,184,592 B, magic MSL2

# ② golden case（PyTorch 基准，取自 tests/golden_cases.jsonl 第 1 条）
python scripts/gen_golden_ts.py   # -> harmony/.../ets/model/GoldenData.ets

# ③ 构建/签名/安装/实机验证
bash harmony/scripts/build_hap.sh     # HAP 41.3MB，BUILD SUCCESSFUL 10s
bash harmony/scripts/sign_hap.sh
hdc install -r harmony/tools/hapsigner/signApp.hap
hdc shell aa start -a EntryAbility -b com.zcode.riskguard
hdc shell "hilog -x | grep -E 'RG_OK|MLP_OK'"
```

## 3. 踩坑与修复

| 现象 | 根因 | 修复 |
|---|---|---|
| `CompileResource` 失败：`Failed to delete ... rawfile/probe_mlp.ms, No error` | WSL 生成的 `.ms` 权限 555 → Windows 只读属性，restool 删不掉中间产物 | `chmod 644` 所有 rawfile 产物 + 清理 `entry/build` 重建 |
| `install` 报 `no such file`（路径拼接成 `C:\...\harmony\/c/Users/...`） | `MSYS_NO_PATHCONV=1` 同时禁掉了本地 HAP 路径转换 | 安装命令改用 Windows 路径（脚本已修正） |

## 4. 产物

| 路径 | 内容 |
|---|---|
| `harmony/entry/src/main/resources/rawfile/risk_guard.ms` | 41.2MB 转换产物（MSL2） |
| `harmony/entry/src/main/ets/model/GoldenData.ets` | 128 个 input_ids / mask + PyTorch logits 基准 |
| `harmony/entry/src/main/ets/pages/Index.ets` | 双探针页面（M2-A MLP + M2-B MiniRBT，自适应 int64/int32） |
| `harmony/scripts/convert_onnx_to_ms.sh` | ONNX→.ms 可复现脚本 |
| `docs/evidence/m2b_real_model_pass.jpeg` | 实机 UI 截图（双 PASS + 数值） |

## 5. 后续（M2-C/D/E/F）

- **M2-C tokenizer**：用 30 条 golden cases 校验端侧 tokenizer（vocab.txt + WordPiece）
- **M2-D Native**：NAPI + `libmindspore_lite_ndk.so` 的 C++ 推理路径（当前为 ArkTS API）
- **M2-E UI**：输入指令/工具/上下文 → 三类分数 + CONTINUE/ASK/BLOCK + 策略提示
- **M2-F 基准**：冷启动、P50/P95、连续 1000 次
