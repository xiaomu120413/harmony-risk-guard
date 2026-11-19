# M1 Runbook — Windows CPU 训练与测试实录

执行日期：2026-11-19 · 环境：Windows 11 / i7-12700（12C20T）/ 31.7 GB RAM / Python 3.14.2
依赖锁定：[requirements-lock.txt](../requirements-lock.txt)（torch 2.15.0+cpu, transformers 4.57.6,
onnx 1.19.1, onnxruntime 1.32.0, scikit-learn 1.7.x, psutil 7.x）

## 0. 复现命令（与实际执行一致）

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\run_all.ps1 -SkipInstall          # 全流程：校验→E0→E1→E2→评测→ONNX→对齐→演示
.\run_all.ps1 -SkipInstall -OnlySteps predict_demo   # 只重跑指定步骤
# 或分步（脚本默认 --model-dir 指向当前胜者 models/finetuned_last2）：
```

> 注：首次全流程运行发生在仓库首次提交之前，training/evaluation 报告中
> `git_commit` 为 null 属预期；此后任何重跑都会记录提交哈希。

## 1. 结果总表（test split，72 条，9 个 held-out 场景组；全部合成数据）

| 模型 | 选择依据 val macro-F1 | test macro-F1 | BLOCK recall | BLOCK 漏判 | 单样本延迟 mean/P50/P95 |
|---|---|---|---|---|---|
| E0 TF-IDF+LogReg | — | 0.783 | 0.667 | 8 | — |
| E1 MiniRBT head-only | 0.702 | 0.689 | **0.750** | **6** | 37.2 / 31.8 / 67.0 ms |
| **E2 MiniRBT last2（胜者）** | **0.870** | 0.778 | 0.667 | 8 | 36.4 / 32.0 / 62.9 ms |

- 模型选择只看 val（E2 0.870 > E1 0.702）；test 从未参与任何选择。
- E2 训练：seed=42，head lr 2e-3 + 解冻末 2 层 lr 2e-5，batch 8，best @ epoch 15/20，
  训练峰值 RSS ≈ 457 MB（`reports/training_last2.json`）。
- 评测峰值内存 457 MB；i7-12700 单样本 P50 ≈ 32 ms。
- 声明：以上指标仅为 36 组合成场景上的表现，**不构成真实业务安全准确率**。

## 2. P0 修复（全部实证）

| 问题 | 证据 | 修复 |
|---|---|---|
| v0.1/v0.2 schema 不兼容 | v0.1 扁平 `instruction/tool/...` vs v0.2 嵌套 `user_instruction/tool_arguments/context`；旧 `render()` 直接 KeyError | `training/common.py`：`normalize_row()` 双格式归一，`render()` 仅消费 canonical 白名单字段，`label/reason/risk_factors/sample_id/group_id` 构造上不可进入输入 |
| run_all.ps1 覆盖 data/ | 旧版第 9 行无条件跑 `make_dataset.py` → 覆写 `data/{train,val,test}.jsonl` | 默认不重建；`make_dataset.py` 写入白名单外目录直接拒绝；`-RegenerateData` 显式触发且先备份到 `synthetic_v01_backup_<ts>` |
| 数据质量 | `reports/data_quality.json`：144/72/72，36 组三方交集空，ID 无重复，同文本无标签冲突，泄漏检查 0 错误 | `validate_dataset.py` 退出码非零即失败 |

## 3. 失败案例与根因（任务书第 9 节要求如实记录）

### 3.1 E1 初训不收敛（train_loss 卡在 ≈1.10 = ln3）
- **现象**：3~7 个 epoch，train_loss 1.10→1.08 几乎不动，val F1 ≈ 0.17~0.33。
- **诊断**（`scripts/diagnose_frozen.py`）：冻结 encoder 的 CLS 特征 + sklearn LR 即可
  train F1=1.0、val≈0.68 → 特征可分，问题在优化与冻结策略。
- **根因**（两条）：
  1. `hfl/minirbt-h256` checkpoint **不含 pooler 权重**（HF 警告列出 `bert.pooler.*`
     为随机初始化），旧代码把 `base_model` 整体冻结 → 随机 pooler 被冻死；
  2. AdamW 每步位移 ~lr，54 步 × lr 5e-4 不足以把新初始化头推到位（v0.1 的
     “epochs 3 + lr 5e-4” 配置在 144 样本下步数严重不足）。
- **修复**：head 模式=冻结 embeddings+encoder，pooler+classifier 可训练；head lr 2e-3；
  E2 双参数组（head 2e-3 / 解冻层 2e-5）；epochs 20 + 早停 patience 5。
- **结果**：E1 val 0.17→0.702；E2 val 0.870。

### 3.2 ONNX 导出进程内 smoke 假报 4.6 logits 差
- **现象**：`export_onnx.py` 内 3 条样本 ORT vs PyTorch diff=4.6，但独立进程
  `validate_onnx.py` 30 条 diff=1.8e-5。
- **诊断**（`scripts/probe_export_diff.py`）：A/B/C/D 四组对照，导出后同实例
  `training=True`（dropout 激活），新实例 vs ORT 仅 1.5e-5。
- **根因**：`torch.onnx.export`（classic tracer, torch 2.15）把模块留在 train 模式，
  进程内对比成了 "dropout forward vs eval 图"。
- **修复**：导出后显式 `model.eval()`；smoke diff 降至 1.5e-5。
- **结论**：导出的 `models/risk_guard.onnx` 本身自始至终是忠实的。

### 3.3 run_all.ps1 的三个 Windows PowerShell 5.1 兼容坑（全部已修复并实测）
1. **stderr 杀死整个脚本**：`$ErrorActionPreference='Stop'` + `2>&1` 管道下，
   python（transformers）写往 stderr 的警告行被转成 ErrorRecord，第一条就终止脚本
   （现象：train 步启动即静默退出，前两步恰好无 stderr）。修复：调用外部命令时
   临时 `EAP=Continue` + 捕获输出 + 显式 `$LASTEXITCODE` 检查。
2. **无 BOM UTF-8 .ps1 被 PS 5.1 按 ANSI/GBK 解析**：`永久删除用户重要资料`
   变成 GBK 乱码传给 predict.py，判定从 BLOCK 漂移到 ASK。修复：.ps1 加 UTF-8 BOM
   + 脚本内 `chcp 65001` + `[Console]::OutputEncoding=UTF8` + `PYTHONIOENCODING=utf-8`。
3. **`Where-Object` 单元素解包**：`-OnlySteps predict_demo` 只匹配一步时管道把
   结果展开成标量，foreach 退化为按字符迭代（`$step[0]='p'`）。修复：`@()` 包裹。

### 3.4 predict 分布外反事实失败（最重要的安全发现）
- **现象**：`--authorization denied --recoverable false`（备份默认 true）的
  file.delete 被判 **CONTINUE(0.99)** ——方向性错误。
- **根因**：v0.2 数据中 file.delete 的 BLOCK 样本全部是
  `authorization=ambiguous + backup=false`；**`denied × file.delete` 组合在训练
  分布中不存在**（denied 只出现在外发类）。模型对未见过的授权状态外推失败。
  分布内同构样本（ambiguous+无备份）正确判 BLOCK(0.64)。
- **影响**：实证任务书第 5 节"反事实对缺失"的风险不是理论——
  **该模型不能作为高危操作的独立闸门，必须配确定性策略引擎并 fail closed**。
  这也验证了任务书第 9 节验收条款。

## 4. M1 验收对照

| 任务 | 产物 | 状态 |
|---|---|---|
| M1-01 环境 | 本文档环境段 + requirements-lock.txt | ✅ |
| M1-02 数据规范化 | `training/common.py` + `reports/data_quality.json` | ✅ |
| M1-03 E0 基线 | `models/baseline_e0.joblib` + `reports/baseline_e0.json` | ✅ |
| M1-04 E1 | `models/finetuned_head` + `reports/training_head.json` | ✅ |
| M1-05 E2 | `models/finetuned_last2` + `reports/training_last2.json` | ✅ |
| M1-06 评测 | `reports/evaluation_e{1,2}_test.json`（含每类指标/混淆矩阵/BLOCK 漏判/延迟） | ✅ |
| M1-07 ONNX | `models/risk_guard.onnx`（[1,128] int64）+ `reports/onnx_export.json` | ✅ |
| M1-08 ORT 对齐 | `reports/onnx_parity.json`：30 条（10/10/10），标签一致率 100%，max diff 1.8e-5；`tests/golden_cases.jsonl` 供 M2 复用 | ✅ |
| M1-09 性能/复现 | 训练/评测报告内含 seed、git commit、峰值 RSS、P50/P95；本文档复现命令 | ✅ |

## 5. 下一步（M2 进度）

1. ~~M2-A 最小穿刺~~ ✅ **已完成（2026-11-19）**：ONNX→`.ms` 转换器可用 +
   鸿蒙 PC（MateBook Pro / API 26）MindSpore Lite 运行时加载执行成功，
   logits 与 PyTorch 一致（max |Δ|=4.292e-7）。详见 [m2a-probe.md](m2a-probe.md)，
   工程见 [../harmony/](../harmony/)，截图 [evidence](evidence/m2a_probe_pass.jpeg)。
2. ~~M2-B 真实模型~~ ✅ **已完成（2026-11-19）**：`risk_guard.onnx`（41.5MB）一次转换成功，
   实机 load 24ms / infer 17ms，logits max |Δ|=7.387e-6，标签一致（BLOCK）；
   converter 自动把 int64 输入降为 int32（端侧自适应填充）。详见
   [m2b-real-model.md](m2b-real-model.md)，截图 [evidence](evidence/m2b_real_model_pass.jpeg)。
3. ~~M2-C Tokenizer~~ ✅ **已完成（2026-11-19）**：纯 ArkTS WordPiece 端侧实现，
   31/31 golden cases token IDs 与 HuggingFace 完全一致（含超长文本截断用例；
   修复 UTF-8 词表解码 bug）。详见 [m2c-tokenizer.md](m2c-tokenizer.md)，
   截图 [evidence](evidence/m2c_all_probes_pass.jpeg)。
4. ~~M2-D Native~~ ✅ **已完成（2026-11-19）**：NAPI + MindSpore Lite C API 实机推理成功
   （17ms/次，logits max|Δ|=7.387e-6）。关键根因：C API 的 Context 必须显式添加
   CPU DeviceInfo，且重建模型时 context 必须成对重建。详见 [m2d-native.md](m2d-native.md)，
   截图 [evidence](evidence/m2d_native_pass.jpeg)。
5. ~~M2-E UI / M2-F 基准~~ ✅ **已完成（2026-11-19）**：交互式判定 UI（输入 → 三类分数
   + 决策 + 策略提示），实测 ASK 99.92%；1000 次连续推理 P50 15ms / P95 16ms，
   Pss ≈99.5MB，无 ANR（分批让出 UI 线程）。详见 [m2ef-ui-bench.md](m2ef-ui-bench.md)，
   截图 [evidence](evidence/m2ef_sandbox_bench.jpeg)。
6. 反事实数据工程（任务书第 5 节）：denied×file.delete 等授权翻转对，重训并复测 case3.4。

## 6. 备选路线调研：cann-local-lm（M3 生成式模型方向）

2026-11-19 在 WSL2 Ubuntu 24.04 上完成真实穿刺，详见
[cann-local-lm-spike.md](cann-local-lm-spike.md) 与 [evidence](evidence/cann-local-lm_spike_evidence.txt)。

- ✅ 环境打通：CANN 9.3.0 mobile-station + KirinX90 ops + P1 venv（torch 2.9.0+cpu）+ torch-delegate + torchair（ABI 20900）
- ✅ **Qwen3.5-0.8B → epair 导出成功**（2.0GB：双图 JSON + 分离权重 + tokenizer + model.json），CPU-only 可跑，无需 NPU
- ❌ **KirinX90 omc 编译被算子覆盖硬阻**：Conv2DV2 / SoftplusV2 / L2Normalize / ChunkGatedDeltaRule 不在该工具链的 kirinx90 ops-info 中（非配置问题）
- 结论：M2 继续 ONNX→MindSpore Lite；M3 上 Kirin 需全量 ops 包或 aarch64 宿主按官方 workarounds 流程
- 过程修复 6 个上游/环境问题（CRLF、wsl 转义层、hccl stubs、master/tag API 不一致、prefill_seq 签名、config 元数据缺失），详见报告
