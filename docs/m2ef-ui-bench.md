# M2-E / M2-F：交互式风险判定 UI + 性能基准 — PASS

日期：2026-11-19 · 设备：HUAWEI MateBook Pro（HAD-W32）/ OpenHarmony 7.1 / API 26
实现：`harmony/entry/src/main/ets/pages/Index.ets`（M2-E 沙盒 + M2-F 基准）

## 1. M2-E：交互式风险判定（离线 UI）

输入：用户指令 / 工具 / 目标类型 / 授权状态（approved·ambiguous·denied）/
可恢复 / 目的地信任。点"运行判定"后走完整离线链路：

```
render() 结构化序列化 → ArkTS BertTokenizer.encode → 原生 predict（C API）
→ softmax → CONTINUE/ASK/BLOCK 三类分数 + 决策 + 策略提示
```

实测（默认示例：`整理下载目录 / file.delete / user_document / ambiguous / 不可恢复 / trusted`）：

```
分数: CONTINUE=0.05%  ASK=99.92%  BLOCK=0.03%
决策: ASK (99.92%)
策略提示：不自动执行，先向用户补充授权/目的地/恢复条件。
```

- 判定方向与训练数据策略一致（删除操作 + 未明确授权 → ASK）
- 策略提示与任务书"模型只给建议、策略引擎兜底"的边界一致

## 2. M2-F：性能与稳定性基准

| 指标 | 结果 |
|---|---|
| 连续推理 | **n=1000**，全部成功 |
| 单次延迟 | **mean 15.0ms / P50 15ms / P95 16ms / min 14ms / max 58ms** |
| 模型加载 | ArkTS API 文件加载 25ms；原生 C API 内存加载 227ms（首载含建图） |
| 进程内存 | Pss ≈ 99.5 MB（`hidumper --mem`） |
| 稳定性 | 1000 次连续运行无失败、无崩溃 |

**踩坑**：初版 1000 次同步循环阻塞 UI 线程约 20s，触发系统"应用无响应"弹窗；
改为**每 25 次让出事件循环**（setTimeout 分批）后，UI 全程响应，基准结果一致。

## 3. 证据

- 截图：[m2ef_sandbox_bench.jpeg](evidence/m2ef_sandbox_bench.jpeg)（M2-D PASS + M2-E 判定 + M2-F 结果同屏）
- 日志：`hilog` 标签 `RiskGuardProbe`（SB_OK / BENCH_OK）
- UI 自动化：harmonyos-dev-mcp（find_elements + click 触发按钮，scroll 后定位）

## 4. M2 总览（A~F）

| 项 | 状态 | 关键结论 |
|---|---|---|
| M2-A 最小穿刺 | ✅ | converter 2.11.0 + MindSporeLiteKit 可用；MLP max|Δ|=4.3e-7 |
| M2-B 真实模型 | ✅ | MiniRBT 41.5MB；load 25ms / infer 17ms；logits max|Δ|=7.4e-6 |
| M2-C Tokenizer | ✅ | ArkTS WordPiece；31/31 golden cases 一致（含截断） |
| M2-D Native | ✅ | NAPI + C API（需 CPU DeviceInfo）；infer 17ms |
| M2-E UI | ✅ | 交互输入 → 三类分数 + 决策 + 策略提示 |
| M2-F 基准 | ✅ | 1000 次连续，P50 15ms / P95 16ms，Pss 99.5MB |
