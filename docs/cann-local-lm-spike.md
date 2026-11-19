# cann-local-lm 转换路线调研（M3 备选路线穿刺）

日期：2026-11-19 · 执行环境：Windows 11 + WSL2 Ubuntu 24.04（x86_64，非 aarch64）
结论摘要：**epair 导出成功（2.0GB 产物），KirinX90 的 omc 编译被算子覆盖硬阻。**

## 1. 这条路线是什么

[cann-local-lm](https://gitcode.com/cann/cann-local-lm) 是华为 CANN 的端侧**生成式大模型**转换+推理引擎：

```
HF 模型 ──cllm-converter──▶ epair（图 JSON + 分离权重 + tokenizer + model.json）
        ──cann_lm_compile──▶ .omc（KirinX90 移动端编译产物）
        ──cann_local_lm_engine──▶ 鸿蒙设备推理
```

- 支持模型：**仅 gemma4 / qwen3.5**（BERT 分类器不支持——本项目 M1 的 MiniRBT 走不了这条线，M2 仍走 ONNX→MindSpore Lite）
- 与本项目的关系：任务书 **M3 生成式模型路线**（Qwen 0.6B~3B）的候选技术栈

## 2. 本次实际完成的工作（全部真实执行）

| 步骤 | 结果 | 证据 |
|---|---|---|
| WSL2 Ubuntu 24.04 环境 | ✅ | gcc 13.3 / Python 3.12.3 / 866G 空闲 |
| CANN 9.3.0 mobile-station toolkit（908MB） | ✅ 安装 | `~/Ascend/cann-9.3.0` |
| Ascend-cann-kirinx90-ops 9.3.0（122MB，含 ops-nn/math/cv/transformer） | ✅ 安装（`--upgrade --quiet`） | kernel 116+25+24+11 个 |
| torch-delegate（master 源码构建） | ✅ | `torch_delegate-0.1.0-py3-none-linux_x86_64.whl` |
| torchair（源码编译，ABI=torch 2.9.0） | ✅ `TORCH_VERSION=20900` | 需手动补 hccl stubs |
| Qwen/Qwen3.5-0.8B（ModelScope LFS） | ✅ 3.3GB | 752,393,024 参数 |
| **epair 导出（prefill+decode）** | ✅ **EXIT=0，2.0GB** | 见下节产物清单 |
| cann_lm_compile（C++ 源码构建） | ✅ | `build/cann_lm_compile` |
| **KirinX90 omc 编译** | ❌ **算子覆盖硬阻** | 见 §4 |

### epair 导出产物（`~/work/epair_qwen35_08b/`，2.0GB）

```
prefill_ep_json_with_ascend_ir.json   13.6 MB   prefill 图（Ascend IR）
decode_ep_json_with_ascend_ir.json    13.3 MB   decode 图
weights_0.bin                         1.57 GB   分离权重（326 unique blobs，356 共享）
host_weights_0.bin                     508 MB   host 侧权重（external embedding）
weight_info.json / host_weights_info.json        权重索引（hash 去重 v2）
model.json                            1.7 KB    运行时元数据（24 层 / kv-cache 2048 / fp16）
tokenizer.json / tokenizer_config.json 12.8 MB
model_config.json                                配置快照
```

## 3. 过程中修掉的 6 个上游/环境问题（根因记录）

| # | 现象 | 根因 | 修复 |
|---|---|---|---|
| 1 | WSL 内 `pull_models.sh: set: pipefail: invalid option name` | Windows 侧 clone 的仓库是 CRLF 换行 | WSL 内重新 clone（LF） |
| 2 | 通过 `wsl.exe -- bash -c "..."` 传递的脚本变量全部为空 | 多层 shell 转义：`\$VAR` 在到达 WSL 前被展开 | 一律用「Windows 写脚本文件 → WSL sed 去 CR → bash 执行」 |
| 3 | `import torchair` → `libhccl.so` / `libhccl_stub.so` not found | mobile-station 包不含 HCCL；torchair 自己构建了 stubs 但未打包进 whl | 把 `deps/torchair/build/stubs/*.so` 加入 `LD_LIBRARY_PATH` |
| 4 | master converter：`get_sample_inputs() got an unexpected keyword argument 'prefill_seq'` | master 分支 saver 与 qwen3_5 wrapper 签名不一致（gemma4 改了 API，qwen3_5 没跟上） | 给 `Qwen3_5ExportWrapper.get_sample_inputs` 增加 `prefill_seq=None` 参数（语义等价：调用点两值相同，base mask 不消费它） |
| 5 | tag `2026_09_22` converter：`LiteStateError: process initialized for Lite EPAIR export; use separate process for Full` | tag converter 走 Full 导出，需要未公开的 `torch-delegate-harmony`（Full 依赖 torch_npu 自带 TorchAir）；master 的 P1 设计是 Lite 路径 | 回到 master + 修 #4（Lite 路径无需 torch_npu，CPU-only 可跑） |
| 6 | `config.json 缺少必填字段: ['text_config.global_head_dim', 'text_config.bos_token_id']` | Qwen3.5-0.8B 发布配置缺这两个字段（仅 runtime model.json 元数据需要，不影响已导出的图） | 补 `global_head_dim=head_dim=256`、`bos_token_id=eos_token_id`（Qwen 系无 BOS 惯例），备份 `config.json.orig` |

## 4. 硬阻断：KirinX90 omc 编译

`cann_lm_compile` 在 GE 的**算子放置（SelectEngine）**阶段失败：

```
Cannot find engine of op name Conv2DV2 type Conv2DV2
Cannot find engine of op name Swish type Swish
Cannot find engine of op name SoftplusV2 type SoftplusV2
Cannot find engine of op name L2Normalize type L2Normalize
Cannot find engine of op name ChunkGatedDeltaRule type ChunkGatedDeltaRule
```

逐一核对 `opp/built-in/op_impl/ai_core/tbe/config/kirinx90/` 的 ops-info：

- ✅ MatMulV3、RmsNorm 等在 `aic-kirinx90-ops-info-nn.json`（220KB）
- ❌ **Conv2DV2 / SoftplusV2 / L2Normalize / ChunkGatedDeltaRule 在全套 ops-info 中不存在**

这不是环境配置问题（Python 算子桥、自定义 pass 均已绕过/加载），而是 **CANN 9.3.0 mobile-station x86_64 的 KirinX90 算子覆盖面不足**：Qwen3.5 的线性注意力（ChunkGatedDeltaRule）、线性卷积（Conv2DV2）、激活（Swish/SoftplusV2）、qk-norm（L2Normalize）都不在支持列表。

仓库自带的 `docs/cann_lm_converter/kirinx90_compile_workarounds.md` 也印证：KirinX90 的 kernel/engine 覆盖比 Ascend910 稀疏，官方验证范围只是 **8 算子的简单网络**（rmsnorm+matmul+mul+add），且需要手工补 L2Normalize 等缺失算子；其成功环境是 **aarch64 宿主 + cann-9.2.0 + 源码构建的三套 ops 包**。

补充验证：改用 `SOC_VERSION=Ascend950DT_9575`（mobile-station 附带的 950 legacy ops）编译，GE 初始化即失败（status 50331649，无 engine 错误）——该工具链不能作为通用回退。

## 5. 结论与建议

1. **cann-local-lm 的转换链路在本机（Windows+WSL x86_64）已打通到 epair**：模型加载、prefill/decode 双图导出、权重分离、tokenizer、运行时元数据全部真实产出。这说明"HF 模型 → epair"一段是 CPU 可完成、无需 NPU 的。
2. **omc（KirinX90）在此工具链上不可达**：原因是目标芯片算子库覆盖，不是配置。错误信息与证据已固化（[evidence](evidence/cann-local-lm_spike_evidence.txt)）。
3. 对 M2（本项目 MiniRBT 分类器 → 鸿蒙）：**继续走 ONNX → MindSpore Lite**，与本路线无关（本工具只支持 LLM）。
4. 对 M3（生成式模型切片）：若真要上 Kirin 设备，需满足其一：
   - 拿到与目标芯片配套的**全量 ops 包**（CANN 正式发布版 / 新版本），重跑 `cann_lm_compile`；
   - 在 **aarch64 Linux/鸿蒙宿主**上按仓库 `kirinx90_compile_workarounds.md` 的完整流程（含源码构建 ops 包）操作；
   - 或选择官方支持列表内的模型（gemma4）验证机制。
5. 下一次行动前需要用户/团队确认：目标鸿蒙设备型号、HarmonyOS 版本、可获取的 CANN/ops 版本（任务书第 20 节待确认项）。

## 6. 复现命令（WSL2 Ubuntu 24.04，x86_64）

```bash
# 1. CANN（mobile station，Kirin 用）
#   toolkit: https://kiri-obs.obs.cn-north-4.myhuaweicloud.com/Cann%20Large%20Model%20Foundation%208.5.0.rc003/Ascend-cann-toolkit_9.3.0_linux-x86_64-mobile-station.run
#   ops:     .../Ascend-cann-kirinx90-ops_9.3.0_linux-x86_64.run
./Ascend-cann-toolkit_9.3.0_linux-x86_64.run --install --install-path=$HOME/Ascend --quiet
./Ascend-cann-kirinx90-ops_9.3.0_linux-x86_64.run --upgrade --install-path=$HOME/Ascend --quiet
ln -s $HOME/Ascend/cann-9.3.0 $HOME/Ascend/cann

# 2. torch-delegate（P1 venv 由 build.sh 自动创建，torch 2.9.0+cpu）
cd ~/work/cann-local-lm && mkdir -p deps && cp -a torch_delegate deps/torch-delegate
cd deps/torch-delegate && ./build.sh && .venv/bin/pip install --no-deps output/package/torch_delegate-*.whl

# 3. torchair（源码编译，注意两个环境变量与 ABI 校验）
cd .. && git clone https://gitcode.com/Ascend/torchair.git && cd torchair
VENV_PY=$(cd ../torch-delegate && pwd)/.venv/bin/python
PYTHON_BIN_PATH=$VENV_PY TARGET_PYTHON_PATH=$VENV_PY bash build.sh -c
cat tools/TORCH_VERSION   # 必须 20900
../torch-delegate/.venv/bin/pip install --no-deps output/torchair-*.whl
# 补 hccl stubs 到 LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(pwd)/build/stubs:$LD_LIBRARY_PATH

# 4. converter（master + prefill_seq 补丁）
pip install --no-deps --no-build-isolation ./scripts/cann_lm_converter

# 5. 模型 + 转换
bash scripts/pull_models.sh --output-dir ~/models Qwen/Qwen3.5-0.8B
#   （补 config.json 的 text_config.global_head_dim / bos_token_id 后）
source ~/Ascend/cann/set_env.sh
TORCH_DEVICE_BACKEND_AUTOLOAD=0 .venv/bin/cllm-converter \
  --hf-path ~/models/Qwen/Qwen3.5-0.8B --output-dir ~/work/epair_qwen35_08b \
  --mode prefill-decode --target-soc KirinX90

# 6. omc（当前在此步失败，见 §4）
source scripts/cann_lm_compile/set_env.sh && export SOC_VERSION=KirinX90
bash scripts/cann_lm_compile/build_cpp.sh
build/cann_lm_compile ~/work/epair_qwen35_08b/prefill_ep_json_with_ascend_ir.json \
  ~/work/epair_qwen35_08b/weight_info.json ~/work/epair_qwen35_08b/prefill/prefill
```

> WSL 内所有辅助脚本与日志：`C:\Users\mu\Desktop\code\model\_research\wsl_scripts\`
> （证据文件由 `save_evidence.sh` 生成）
