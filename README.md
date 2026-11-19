# Harmony Risk Guard v0.2 — Windows CPU training pipeline (M1)

风险判别模型：输入「用户指令 + 工具 + 参数 + 上下文」，输出 `CONTINUE / ASK / BLOCK` 建议。
本仓库为任务书 M1 阶段的工程实现：v0.1 工程修复版 + v0.2 结构化数据。

## P0 修复（相对 v0.1）

- **v0.1/v0.2 schema 不兼容**：`training/common.py` 统一读取两种格式，规范化为
  canonical 字段后序列化；`label`、`reason`、`risk_factors`、`sample_id`、`group_id`
  等泄漏字段在构造上不可能进入模型输入。
- **run_all.ps1 覆盖 data/ 的隐患**：默认不重建任何数据；`make_dataset.py` 只允许写
  `data/synthetic_v01/`（legacy 演示数据），写入工作目录会被拒绝；
  `-RegenerateData` 显式触发且先备份。
- **新增 `training/validate_dataset.py`**：schema/枚举校验、group_id 跨split交集、
  重复 ID、冲突标签、泄漏检查；报告输出 `reports/data_quality.json`。

## 数据（v0.2，data/）

288 条合成样本（train 144 / val 72 / test 72），3 场景 × 12 基础场景 × 8 文本变体，
按 `group_id` 分组切分（36 组，三方交集为空）。
- `data/{train,val,test}.jsonl` — 工作数据（静态输入，脚本不再生成）
- `data/v0.2_original/` — 数据包原始副本（provenance）
- `data/v0.1_legacy/` — v0.1 旧合成数据（仅保留参考）
- `data/synthetic_v01/` — make_dataset.py 显式生成的 legacy 演示数据（默认不存在）

## 环境

- Windows 10/11 x64；实测 Python 3.14.2 / i7-12700（12C20T）/ 31.7 GB RAM
- 仅 CPU；无需 GPU；首次运行需联网装依赖和下载 `hfl/minirbt-h256`
- PyPI 慢时用清华镜像（run_all.ps1 已内置）

## 实测结果（2026-11-19，test split 72 条，全合成数据）

| 模型 | val macro-F1（选择依据） | test macro-F1 | BLOCK recall / 漏判 |
|---|---|---|---|
| E0 TF-IDF+LogReg | — | 0.783 | 0.667 / 8 |
| E1 head-only | 0.702 | 0.689 | 0.750 / 6 |
| **E2 last2（胜者）** | **0.870** | 0.778 | 0.667 / 8 |

ONNX 对齐：30 条分层样本标签一致率 100%，max logits diff 1.8e-5。
细节、失败案例与根因见 [docs/runbook.md](docs/runbook.md)。

## 运行

```powershell
# 一键全流程（venv + 依赖 + 校验 + E0/E1/E2 + 评测 + ONNX + 对齐 + 演示）
Set-ExecutionPolicy -Scope Process Bypass
.\run_all.ps1

# 20 条样本 smoke test（写到 models/smoke，不碰正式模型目录）
.\run_all.ps1 -Smoke

# 显式重建 legacy 演示数据（先备份；不会动 v0.2 工作数据）
.\run_all.ps1 -RegenerateData
```

或手动分步：

```powershell
python training\validate_dataset.py --data-dir data
python training\baseline.py                     # E0: TF-IDF + LogReg
python training\train.py --device cpu --mode head --epochs 20 --early-stop-patience 5 --threads 8    # E1
python training\train.py --device cpu --mode last2 --epochs 20 --early-stop-patience 5 --threads 8   # E2
python training\evaluate.py --model-dir models\finetuned_head --tag e1 --split test
python training\evaluate.py --model-dir models\finetuned_last2 --tag e2 --split test
python training\export_onnx.py --model-dir models\finetuned_last2 --seq-len 128
python training\validate_onnx.py --model-dir models\finetuned_last2 --samples 30
python training\predict.py --instruction "整理下载目录" --tool file.delete --target "用户文档"
```

## 产物

| 路径 | 内容 |
|---|---|
| `models/finetuned_head` / `finetuned_last2` | E1（分类头）/ E2（解冻末 2 层）最优 checkpoint |
| `models/baseline_e0.joblib` | TF-IDF + LogReg 基线 |
| `models/risk_guard.onnx` | 固定形状 `[1,128]` int64 导出 |
| `reports/*.json` | 数据质量、E0/E1/E2 训练与评测、ONNX 导出与对齐报告 |
| `tests/golden_cases.jsonl` | 30 条分层样本（token IDs + 双端 logits），供 M2 鸿蒙端 parity 复用 |
| `reports/logs/` | run_all 每步完整终端日志 |

## 边界与限制（必读）

- 288 条全部为规则生成的**合成样本**；所有指标仅代表在该合成分布上的表现，
  **不得宣称真实业务准确率**；`reason` 字段不进入模型输入。
- test split 从未用于模型选择；模型选择只看 val。
- ONNX int64 输入对 MindSpore Lite 的兼容性**未验证**；M2 需要独立转换与实机穿刺。
- 模型输出只是风险建议：权限检查、不可恢复删除、凭据外发必须有独立策略引擎兜底，
  不确定时 fail closed。禁止将本模型作为高危操作的唯一闸门。

## 相关调研与进展

- [docs/runbook.md](docs/runbook.md) — M1 全部实测记录、失败案例与复现命令
- [docs/m2a-probe.md](docs/m2a-probe.md) — **M2-A 实机穿刺 PASS**：ONNX→MindSpore Lite `.ms`
  在鸿蒙 PC（MateBook Pro / API 26）加载执行，logits 与 PyTorch 一致（max |Δ|=4.3e-7）
- [docs/m2b-real-model.md](docs/m2b-real-model.md) — **M2-B 真实模型 PASS**：MiniRBT 分类器
  （41.5MB）实机推理 load 24ms / infer 17ms，logits max |Δ|=7.4e-6，标签一致
- [docs/m2c-tokenizer.md](docs/m2c-tokenizer.md) — **M2-C Tokenizer PASS**：纯 ArkTS WordPiece
  端侧实现，30/30 golden cases token IDs 与 HuggingFace 一致
- [docs/m2d-native.md](docs/m2d-native.md) — **M2-D Native PASS**：NAPI + MindSpore Lite C API
  实机推理 17ms/次，logits max |Δ|=7.4e-6（Context 必须加 CPU DeviceInfo）
- [docs/m2ef-ui-bench.md](docs/m2ef-ui-bench.md) — **M2-E/F PASS**：交互式判定 UI
  （ASK 99.92%），1000 次连续推理 P50 15ms / P95 16ms，Pss ≈99.5MB
- [harmony/](harmony/) — 可构建的 HarmonyOS 工程（ArkTS + MindSporeLiteKit + 探针模型）
- [docs/cann-local-lm-spike.md](docs/cann-local-lm-spike.md) — M3 备选路线（CANN 端侧 LLM 转换）穿刺：
  WSL2 上 Qwen3.5-0.8B → epair 导出成功（2.0GB），KirinX90 omc 编译被算子覆盖硬阻（含证据）
