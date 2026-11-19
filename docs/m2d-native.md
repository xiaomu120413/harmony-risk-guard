# M2-D：Native NAPI + MindSpore Lite C API 推理 — PASS

日期：2026-11-19 · 设备：HUAWEI MateBook Pro（HAD-W32）/ OpenHarmony 7.1 / API 26
实现：`harmony/entry/src/main/cpp/riskguard_native.cpp`（NAPI 模块，链接 `libmindspore_lite_ndk.so`）

## 1. 结论

**NAPI 原生模块 + MindSpore Lite C API 在实机上完成模型加载与推理，数值与 PyTorch / ArkTS 路径一致：**

```
ND_DIAG mlp=[0,0,0,0] rg=[0,0,0,0]
input[0] dtype=34 elems=128
input[1] dtype=34 elems=128
predict ok out_elems=3
ND_OK logits=[-1.752675, -3.063437, 4.612434] maxdiff=7.387e-6
      native C API | buffer | load=210ms infer=17ms
```

- 原生 C API 推理 **17ms/次**，与 ArkTS Kit API（17~23ms）持平
- UI 同屏显示 M2-A/B/C/D 四项 PASS（[截图](evidence/m2d_native_pass.jpeg)）

## 2. 关键根因（踩坑记录）

初次接入 C API 时，`OH_AI_ModelBuildFromFile` / `OH_AI_ModelBuild` **无论 file/buffer、
有无 threadNum 配置、MLP/MiniRBT 哪个模型，全部返回 -2（LITE_NULLPTR）**。
矩阵诊断（`diagnose()`：2 模型 × buffer/file × threadNum/default）确认是 API 级问题而非模型问题。

**根因：MindSpore Lite C API 的 Context 必须显式添加 CPU DeviceInfo**：

```cpp
OH_AI_ContextHandle ctx = OH_AI_ContextCreate();
OH_AI_DeviceInfoHandle cpu = OH_AI_DeviceInfoCreate(OH_AI_DEVICETYPE_CPU); // 关键！
OH_AI_ContextAddDeviceInfo(ctx, cpu);
```

补上后诊断矩阵全部返回 0。第二个坑：**重建模型时 context 必须与 model 成对重建**
（仅销毁 model、复用旧 context 会返回 -3 PARAM_INVALID）。

## 3. 实现

```cpp
// 加载：内存路径优先（ArkTS 读 rawfile 后传 ArrayBuffer）
OH_AI_ModelBuild(model, data, size, OH_AI_MODELTYPE_MINDIR, ctx);
// 推理：按输入张量实际 dtype 填充（converter 将 int64 降为 int32，dtype=34）
OH_AI_TensorSetData(t, buf);
OH_AI_ModelPredict(model, inputs, &outputs, nullptr, nullptr);
```

- NAPI 导出：`loadModel(path)` / `loadModelFromBuffer(buf)` / `predict(ids, mask)` / `diagnose(buf, path)`
- 类型声明：`entry/src/main/cpp/types/libriskguard_native/`
- CMake：链接 `libace_napi.z.so` / `libhilog_ndk.z.so` / `libmindspore_lite_ndk.so`

## 4. 产物

| 路径 | 内容 |
|---|---|
| `harmony/entry/src/main/cpp/riskguard_native.cpp` | NAPI + C API 实现（含诊断矩阵） |
| `harmony/entry/src/main/cpp/CMakeLists.txt` | 原生构建配置 |
| `harmony/entry/src/main/cpp/types/libriskguard_native/` | ArkTS 类型声明 |
| `harmony/entry/src/main/ets/pages/Index.ets` | M2-D 探针（buffer 优先、file 回退） |
| `docs/evidence/m2d_native_pass.jpeg` | 四项 PASS 同屏截图 |

## 5. 剩余（M2-E/F）

- **M2-E**：输入指令/上下文 → 三类分数 + CONTINUE/ASK/BLOCK + 策略提示的完整 UI
- **M2-F**：冷启动、P50/P95、连续 1000 次稳定性
