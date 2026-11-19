# 推理后端调研：CPU / NPU / GPU（HarmonyOS PC）

日期：2026-11-19 · 设备：HUAWEI MateBook Pro（HAD-W32）· OpenHarmony 7.1 / API 26
设备芯片：KirinX90 v2（NPU 可用）· 系统推理引擎：**MindSpore Lite 2.7.0**

## 1. 结论

**当前产品固定使用 CPU 推理（单次 15–20ms）。NPU 存在但当前运行时/公开 API 路径无法运行标准 `.ms`。**

| 后端 | 状态 | 说明 |
|---|---|---|
| CPU | ✅ 生产使用 | 三端数值一致（PyTorch/ArkTS/NAPI，max\|Δ\|=7.4e-6）；1000 次 P50 15ms |
| NPU | ❌ 平台限制 | 设备枚举到 `NPU_ohos.boot.hardware.KirinX90_v2_0` 与 `HIAI_F`，但 MS Lite 2.7.0 的两条路径均拒绝标准模型（见 §3） |
| GPU | ❌ 未暴露 | `OH_AI_GetAllNNRTDeviceDescs` 未返回 GPU 加速器 |

## 2. 实测证据

推理后端自检（诊断页 / 无头自动测试）返回：

```
available NNRT devices: NPU_ohos.boot.hardware.KirinX90_v2_0:ACCELERATOR | HIAI_F:ACCELERATOR
BACKEND CPU (threads=2)
```

NPU 尝试的完整失败链（hilog，MS Lite 2.7.0）：

```
Disable heterogeneous inference in Delegate.
[nnrt_delegate.cc:171] CheckNPUPrefix# strncmp failed, device_name: HIAI_F
[nnrt_delegate.cc:237] BuildOfflineModel# not third party model
[scheduler.cc:542] ReplaceDelegateKernels# Delegate prepare kernels failed.
[lite_session.cc:616] CompileGraph# Schedule kernels failed: -1
→ 自动回退 CPU（build status=0）
```

## 3. 根因（对照 MindSpore Lite 源码）

`nnrt_delegate.cc` 的委托构建逻辑（OpenHarmony 镜像源码）：
- **在线 Kirin NPU 路径**：`CheckNPUPrefix("NPU_")` 通过后走 `BuildKirinNPUModel`
- **离线 HiAI Foundation 路径**：`CheckNPUPrefix("HIAI_F")` 通过后走 `BuildOfflineModel`，
  而 `BuildOfflineModel` 第一步要求 `IsCustomModel()`——模型必须是**单个 Custom 节点且最后
  一个输入是预编译模型缓冲（OM/HiAI）**，否则报 `not third party model` 并失败

实测结果：
- 选 `HIAI_F` 设备 → 进入离线路径 → 被 `not third party model` 拒绝（我们是标准 201 节点图）
- 选 `NPU_ohos.boot.hardware.KirinX90_v2_0` 设备 → 两个前缀校验均失败（该运行时在线路径
  不识别此设备名）

**外部交叉验证**（AmphionRuntime 在 Kirin 9020 / HarmonyOS 7.0 的公开记录）：
- HiAI 路径的模型认证（DLSA）按结构触发，transformer block 会强制进入模型认证，个人开发者
  无法自助申请
- `LayerNormalization` 算子在该系 NPU 的 NNRt 编译路径上不被接受（不带注意力的 MatMul+LN 也
  会被秒拒），而 MiniRBT 含 LayerNorm
- 结论与本次实测一致：**NPU 需要厂商预编译/认证的模型包，不是配置问题**

## 4. 版本对齐修复

首次尝试时设备日志提示：

```
The current model version MindSpore Lite 2.11.0 is later than the inference engine
version MindSpore Lite 2.7.0. Use a converter tool whose version is earlier than or
equal to the inference engine version.
```

已用 **MindSpore Lite 2.7.0 converter** 重新转换 `risk_guard.onnx` 与 `probe_mlp.onnx`
（`CONVERT RESULT SUCCESS:0`），替换 rawfile 后版本警告消失，CPU 推理数值与 2.11 产物
逐位一致（决策/分数/latency 不变）。

## 5. CPU 调优实测（可执行方案）

`loadModelFromBufferCfg` 支持线程数/算子并行配置；无头测试每组 30 次推理（独立进程）：

| 线程 | 并行 | avg | p50 | min |
|---|---|---|---|---|
| 2 | off/on | 15ms | 15ms | 14ms |
| **4** | off/on | **9ms** | 9ms | 8ms |
| **8** | off/on | **9ms** | **8ms** | **6ms** |

结论：**4–8 线程比默认 2 线程快约 40%**；`SetEnableParallel` 影响很小。产品默认已改为
**4 线程**（9ms 级别，兼顾 UI 线程竞争）。8 线程可作为高负载场景选项。

工程注意：MS Lite 2.7.0 NDK 桥在**同进程重复 `OH_AI_ModelDestroy` 后重建**时会出现
double free（崩溃栈：`OH_AI_ModelDestroy → LiteSession::~LiteSession → Tensor::FreeData`，
SIGTRAP）。因此性能对比采用"一组配置一个进程"（`--ps autotest perf --pi threads N`），
产品路径保持单次加载，不受影响。

## 6. 复现

```bash
# 无头自检（不依赖 UI 焦点，适合 CI/远程）
hdc shell "aa start -a EntryAbility -b com.zcode.riskguard --ps autotest npu"
hdc shell "hilog -x | grep -E 'NPU device info|CheckNPUPrefix|BuildOfflineModel|NPU_LOAD'"

# 评估矩阵（4 组固定输入，验证模型不是恒定输出）
hdc shell "aa start -a EntryAbility -b com.zcode.riskguard --ps autotest assess"

# 设备可用加速器枚举（App 诊断页也会显示）
#   RiskGuardNative: available NNRT devices: ...
```

## 7. 若未来要走 NPU

1. 以企业主体通过华为接口人申请 **DLSA 模型认证**（模型级认证，非应用签名）
2. 使用厂商 HiAI/OM 预编译链路生成"第三方模型"包（Custom 节点 + OM 缓冲），或等运行时
   开放在线 NPU 路径的设备名
3. 解决 `LayerNormalization` 在目标 NPU 的表达（外部实测显示拆成基础算子也会被更早拒绝）
4. 通过后需重新做精度对齐（NPU fp16 与 CPU fp32 的 logits 容差需单独记录）
