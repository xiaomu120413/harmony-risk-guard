#include <napi/native_api.h>
#include <hilog/log.h>
#include <mindspore/model.h>
#include <mindspore/context.h>
#include <mindspore/tensor.h>
#include <mindspore/data_type.h>
#include <mindspore/types.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#define LOG_TAG "RiskGuardNative"
#define LOGI(...) OH_LOG_Print(LOG_APP, LOG_INFO, 0xD001, LOG_TAG, __VA_ARGS__)
#define LOGE(...) OH_LOG_Print(LOG_APP, LOG_ERROR, 0xD001, LOG_TAG, __VA_ARGS__)

static OH_AI_ModelHandle g_model = nullptr;
static OH_AI_ContextHandle g_ctx = nullptr;
static std::string g_backendDesc = "";  // e.g. "CPU (threads=2)"

// Collect NNRT accelerators, preferring the HiAI Foundation device ("HIAI_F"),
// which is the path MindSpore Lite 2.7 uses for offline Kirin NPU inference.
struct AccelInfo {
    std::string name;
    size_t deviceId;
};

static std::vector<AccelInfo> FindAccelerators()
{
    std::vector<AccelInfo> out;
    size_t num = 0;
    NNRTDeviceDesc *descs = OH_AI_GetAllNNRTDeviceDescs(&num);
    if (descs != nullptr) {
        for (size_t i = 0; i < num; i++) {
            NNRTDeviceDesc *d = OH_AI_GetElementOfNNRTDeviceDescs(descs, i);
            if (d == nullptr) {
                continue;
            }
            if (OH_AI_GetTypeFromNNRTDeviceDesc(d) != OH_AI_NNRTDEVICE_ACCELERATOR) {
                continue;
            }
            const char *n = OH_AI_GetNameFromNNRTDeviceDesc(d);
            if (n != nullptr) {
                AccelInfo info;
                info.name = n;
                info.deviceId = OH_AI_GetDeviceIdFromNNRTDeviceDesc(d);
                out.push_back(info);
            }
        }
        OH_AI_DestroyAllNNRTDeviceDescs(&descs);
    }
    // Preference: HIAI_F first, then NPU_* (online Kirin path), then others.
    std::vector<AccelInfo> ordered;
    for (size_t i = 0; i < out.size(); i++) {
        if (out[i].name.rfind("HIAI_F", 0) == 0) {
            ordered.push_back(out[i]);
        }
    }
    for (size_t i = 0; i < out.size(); i++) {
        if (out[i].name.rfind("NPU_", 0) == 0) {
            ordered.push_back(out[i]);
        }
    }
    if (ordered.empty()) {
        ordered = out;
    }
    return ordered;
}

// Standard MindSpore Lite C API setup: a context must carry a device info.
// npuMode: 0=CPU, 1=NNRT accel (HIAI_F first), 2=NNRT online (NPU_ first),
//          3=KIRIN_NPU device type, 4=KIRIN_NPU + CPU (heterogeneous).
static bool EnsureContext(int threadNum, int npuMode)
{
    if (g_ctx != nullptr) {
        return true;
    }
    g_ctx = OH_AI_ContextCreate();
    if (g_ctx == nullptr) {
        LOGE("ContextCreate failed");
        return false;
    }
    if (npuMode == 3 || npuMode == 4) {
        OH_AI_DeviceInfoHandle npu = OH_AI_DeviceInfoCreate(OH_AI_DEVICETYPE_KIRIN_NPU);
        if (npu != nullptr) {
            OH_AI_DeviceInfoSetPerformanceMode(npu, OH_AI_PERFORMANCE_HIGH);
            OH_AI_ContextAddDeviceInfo(g_ctx, npu);
            g_backendDesc = "KIRIN_NPU (device type)";
            LOGI("KIRIN_NPU device info added");
            if (npuMode == 4) {
                OH_AI_DeviceInfoHandle cpu = OH_AI_DeviceInfoCreate(OH_AI_DEVICETYPE_CPU);
                if (cpu != nullptr) {
                    OH_AI_ContextAddDeviceInfo(g_ctx, cpu);
                    g_backendDesc += " + CPU";
                    LOGI("CPU device info added after KIRIN_NPU");
                }
            }
            if (threadNum > 0) {
                OH_AI_ContextSetThreadNum(g_ctx, threadNum);
            }
            return true;
        }
        LOGE("KIRIN_NPU device info unavailable");
    }
    if (npuMode == 1 || npuMode == 2) {
        std::vector<AccelInfo> accels = FindAccelerators();
        if (npuMode == 2) {
            std::vector<AccelInfo> online;
            for (size_t i = 0; i < accels.size(); i++) {
                if (accels[i].name.rfind("NPU_", 0) == 0) {
                    online.push_back(accels[i]);
                }
            }
            if (!online.empty()) {
                accels = online;
            }
        }
        if (!accels.empty()) {
            const AccelInfo &pick = accels[0];
            OH_AI_DeviceInfoHandle npu = OH_AI_CreateNNRTDeviceInfoByName(pick.name.c_str());
            if (npu == nullptr) {
                npu = OH_AI_CreateNNRTDeviceInfoByType(OH_AI_NNRTDEVICE_ACCELERATOR);
            }
            if (npu != nullptr) {
                OH_AI_DeviceInfoSetDeviceId(npu, pick.deviceId);
                OH_AI_DeviceInfoSetPerformanceMode(npu, OH_AI_PERFORMANCE_HIGH);
                OH_AI_ContextAddDeviceInfo(g_ctx, npu);
                g_backendDesc = "NPU (" + pick.name + ", id=" + std::to_string(pick.deviceId) + ")";
                LOGI("NPU device info added: %{public}s", g_backendDesc.c_str());
                if (threadNum > 0) {
                    OH_AI_ContextSetThreadNum(g_ctx, threadNum);
                }
                return true;
            }
        }
        LOGI("NPU device info unavailable, falling back to CPU");
    }
    OH_AI_DeviceInfoHandle cpu = OH_AI_DeviceInfoCreate(OH_AI_DEVICETYPE_CPU);
    if (cpu == nullptr) {
        LOGE("DeviceInfoCreate(CPU) failed");
        return false;
    }
    OH_AI_ContextAddDeviceInfo(g_ctx, cpu);
    g_backendDesc = threadNum > 0
        ? "CPU (threads=" + std::to_string(threadNum) + ")"
        : "CPU (default threads)";
    LOGI("CPU device info added: %{public}s", g_backendDesc.c_str());
    if (threadNum > 0) {
        OH_AI_ContextSetThreadNum(g_ctx, threadNum);
    }
    return true;
}

static bool GetStringArg(napi_env env, napi_value v, std::string *out)
{
    size_t len = 0;
    if (napi_get_value_string_utf8(env, v, nullptr, 0, &len) != napi_ok) {
        return false;
    }
    out->resize(len);
    size_t copied = 0;
    return napi_get_value_string_utf8(env, v, out->data(), len + 1, &copied) == napi_ok;
}

static bool GetIntArray(napi_env env, napi_value v, std::vector<int32_t> *out)
{
    uint32_t len = 0;
    if (napi_get_array_length(env, v, &len) != napi_ok) {
        return false;
    }
    out->resize(len);
    for (uint32_t i = 0; i < len; i++) {
        napi_value el = nullptr;
        if (napi_get_element(env, v, i, &el) != napi_ok) {
            return false;
        }
        int32_t x = 0;
        if (napi_get_value_int32(env, el, &x) != napi_ok) {
            return false;
        }
        (*out)[i] = x;
    }
    return true;
}

static napi_value LoadModel(napi_env env, napi_callback_info info)
{
    size_t argc = 1;
    napi_value argv[1] = {nullptr};
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);
    std::string path;
    if (argc < 1 || !GetStringArg(env, argv[0], &path)) {
        LOGE("loadModel: bad path argument");
        return nullptr;
    }
    if (g_model != nullptr) {
        OH_AI_ModelDestroy(&g_model);
        g_model = nullptr;
    }
    if (g_ctx != nullptr) {
        OH_AI_ContextDestroy(&g_ctx);
        g_ctx = nullptr;
    }
    if (!EnsureContext(4, 0)) {
        napi_value err = nullptr;
        napi_create_int32(env, -1000, &err);
        return err;
    }
    g_model = OH_AI_ModelCreate();
    LOGI("ModelCreate -> %{public}s", g_model == nullptr ? "NULL" : "ok");
    if (g_model == nullptr) {
        napi_value err = nullptr;
        napi_create_int32(env, -1001, &err);
        return err;
    }
    OH_AI_Status st = OH_AI_ModelBuildFromFile(g_model, path.c_str(), OH_AI_MODELTYPE_MINDIR, g_ctx);
    LOGI("ModelBuildFromFile path=%{public}s status=%{public}d", path.c_str(), static_cast<int>(st));
    napi_value result = nullptr;
    napi_create_int32(env, static_cast<int32_t>(st), &result);
    return result;
}

static napi_value LoadModelFromBuffer(napi_env env, napi_callback_info info)
{
    size_t argc = 1;
    napi_value argv[1] = {nullptr};
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);
    void *data = nullptr;
    size_t len = 0;
    bool isAb = false;
    napi_is_arraybuffer(env, argv[0], &isAb);
    if (isAb) {
        napi_get_arraybuffer_info(env, argv[0], &data, &len);
    } else {
        bool isTa = false;
        napi_is_typedarray(env, argv[0], &isTa);
        if (isTa) {
            napi_typedarray_type type;
            size_t length = 0;
            void *td = nullptr;
            napi_value ab = nullptr;
            size_t offset = 0;
            napi_get_typedarray_info(env, argv[0], &type, &length, &td, &ab, &offset);
            data = td;
            len = length;
        }
    }
    if (data == nullptr || len == 0) {
        LOGE("loadModelFromBuffer: bad buffer");
        napi_value err = nullptr;
        napi_create_int32(env, -1002, &err);
        return err;
    }
    if (g_model != nullptr) {
        OH_AI_ModelDestroy(&g_model);
        g_model = nullptr;
    }
    if (g_ctx != nullptr) {
        OH_AI_ContextDestroy(&g_ctx);
        g_ctx = nullptr;
    }
    if (!EnsureContext(4, 0)) {
        napi_value err = nullptr;
        napi_create_int32(env, -1003, &err);
        return err;
    }
    g_model = OH_AI_ModelCreate();
    if (g_model == nullptr) {
        napi_value err = nullptr;
        napi_create_int32(env, -1003, &err);
        return err;
    }
    OH_AI_Status st = OH_AI_ModelBuild(g_model, data, len, OH_AI_MODELTYPE_MINDIR, g_ctx);
    LOGI("ModelBuild size=%{public}zu status=%{public}d", len, static_cast<int>(st));
    napi_value result = nullptr;
    napi_create_int32(env, static_cast<int32_t>(st), &result);
    return result;
}

// Diagnostic: try build strategies for a model and return status array.
// args: buffer, path, useThreadNum(boolean) -> returns [buf_with_tn, buf_default, file_with_tn, file_default]
static OH_AI_Status TryBuild(void *data, size_t len, const char *path, bool fromFile, int threadNum)
{
    if (g_model != nullptr) {
        OH_AI_ModelDestroy(&g_model);
        g_model = nullptr;
    }
    if (g_ctx != nullptr) {
        OH_AI_ContextDestroy(&g_ctx);
        g_ctx = nullptr;
    }
    if (!EnsureContext(threadNum, 0)) {
        return OH_AI_STATUS_LITE_NULLPTR;
    }
    g_model = OH_AI_ModelCreate();
    if (g_model == nullptr) {
        return OH_AI_STATUS_LITE_NULLPTR;
    }
    if (fromFile) {
        return OH_AI_ModelBuildFromFile(g_model, path, OH_AI_MODELTYPE_MINDIR, g_ctx);
    }
    return OH_AI_ModelBuild(g_model, data, len, OH_AI_MODELTYPE_MINDIR, g_ctx);
}

static napi_value Diagnose(napi_env env, napi_callback_info info)
{
    size_t argc = 2;
    napi_value argv[2] = {nullptr, nullptr};
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);
    void *data = nullptr;
    size_t len = 0;
    bool isAb = false;
    napi_is_arraybuffer(env, argv[0], &isAb);
    if (isAb) {
        napi_get_arraybuffer_info(env, argv[0], &data, &len);
    } else {
        bool isTa = false;
        napi_is_typedarray(env, argv[0], &isTa);
        if (isTa) {
            napi_typedarray_type type;
            size_t length = 0;
            void *td = nullptr;
            napi_value ab = nullptr;
            size_t offset = 0;
            napi_get_typedarray_info(env, argv[0], &type, &length, &td, &ab, &offset);
            data = td;
            len = length;
        }
    }
    std::string path;
    if (argc < 2 || !GetStringArg(env, argv[1], &path)) {
        path = "";
    }
    OH_AI_Status st[4];
    st[0] = TryBuild(data, len, path.c_str(), false, 2);
    LOGI("diagnose buf+tn => %{public}d", static_cast<int>(st[0]));
    st[1] = TryBuild(data, len, path.c_str(), false, 0);
    LOGI("diagnose buf+default => %{public}d", static_cast<int>(st[1]));
    st[2] = TryBuild(data, len, path.c_str(), true, 2);
    LOGI("diagnose file+tn => %{public}d", static_cast<int>(st[2]));
    st[3] = TryBuild(data, len, path.c_str(), true, 0);
    LOGI("diagnose file+default => %{public}d", static_cast<int>(st[3]));
    // leave the model in the best state: if any succeeded keep last success, else last attempt
    napi_value arr = nullptr;
    napi_create_array_with_length(env, 4, &arr);
    for (uint32_t i = 0; i < 4; i++) {
        napi_value num = nullptr;
        napi_create_int32(env, static_cast<int32_t>(st[i]), &num);
        napi_set_element(env, arr, i, num);
    }
    return arr;
}

// Report which compute backend the model was built for (we only ever add a CPU
// device info), plus which NNRT accelerators the system exposes (for reference).
static const char *NnrtTypeName(OH_AI_NNRTDeviceType t)
{
    switch (t) {
        case OH_AI_NNRTDEVICE_CPU: return "CPU";
        case OH_AI_NNRTDEVICE_GPU: return "GPU";
        case OH_AI_NNRTDEVICE_ACCELERATOR: return "ACCELERATOR";
        default: return "OTHERS";
    }
}

static napi_value InferenceBackend(napi_env env, napi_callback_info info)
{
    std::string out = g_backendDesc;
    if (out.empty()) {
        out = "CPU (context not created yet)";
    }
    napi_value res = nullptr;
    napi_create_string_utf8(env, out.c_str(), out.size(), &res);
    return res;
}

static napi_value AvailableAccelerators(napi_env env, napi_callback_info info)
{
    size_t num = 0;
    NNRTDeviceDesc *descs = OH_AI_GetAllNNRTDeviceDescs(&num);
    std::string out;
    if (descs != nullptr) {
        for (size_t i = 0; i < num; i++) {
            NNRTDeviceDesc *d = OH_AI_GetElementOfNNRTDeviceDescs(descs, i);
            if (d == nullptr) {
                continue;
            }
            const char *name = OH_AI_GetNameFromNNRTDeviceDesc(d);
            OH_AI_NNRTDeviceType t = OH_AI_GetTypeFromNNRTDeviceDesc(d);
            if (!out.empty()) {
                out += " | ";
            }
            out += (name != nullptr ? name : "unknown");
            out += ":";
            out += NnrtTypeName(t);
        }
        OH_AI_DestroyAllNNRTDeviceDescs(&descs);
    }
    if (out.empty()) {
        out = "none";
    }
    LOGI("available NNRT devices: %{public}s", out.c_str());
    napi_value res = nullptr;
    napi_create_string_utf8(env, out.c_str(), out.size(), &res);
    return res;
}

// Build the model with an explicit backend preference ("npu" or "cpu").
// NPU build failure falls back to CPU automatically; the return value is the
// first-attempt status (0 = success), and inferenceBackend() reports the final one.
static napi_value LoadModelWithBackend(napi_env env, napi_callback_info info)
{
    size_t argc = 2;
    napi_value argv[2] = {nullptr, nullptr};
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);
    void *data = nullptr;
    size_t len = 0;
    bool isAb = false;
    napi_is_arraybuffer(env, argv[0], &isAb);
    if (isAb) {
        napi_get_arraybuffer_info(env, argv[0], &data, &len);
    } else {
        bool isTa = false;
        napi_is_typedarray(env, argv[0], &isTa);
        if (isTa) {
            napi_typedarray_type type;
            size_t length = 0;
            void *td = nullptr;
            napi_value ab = nullptr;
            size_t offset = 0;
            napi_get_typedarray_info(env, argv[0], &type, &length, &td, &ab, &offset);
            data = td;
            len = length;
        }
    }
    std::string backend;
    if (argc >= 2) {
        GetStringArg(env, argv[1], &backend);
    }
    int npuMode = 0;
    if (backend == "npu") {
        npuMode = 1;
    } else if (backend == "online") {
        npuMode = 2;
    } else if (backend == "kirin") {
        npuMode = 3;
    } else if (backend == "kirin_het") {
        npuMode = 4;
    }
    napi_value result = nullptr;
    if (data == nullptr || len == 0) {
        napi_create_int32(env, -1002, &result);
        return result;
    }
    if (g_model != nullptr) {
        OH_AI_ModelDestroy(&g_model);
        g_model = nullptr;
    }
    if (g_ctx != nullptr) {
        OH_AI_ContextDestroy(&g_ctx);
        g_ctx = nullptr;
    }
    if (!EnsureContext(4, npuMode)) {
        napi_create_int32(env, -1003, &result);
        return result;
    }
    g_model = OH_AI_ModelCreate();
    if (g_model == nullptr) {
        napi_create_int32(env, -1003, &result);
        return result;
    }
    OH_AI_Status st = OH_AI_ModelBuild(g_model, data, len, OH_AI_MODELTYPE_MINDIR, g_ctx);
    LOGI("LoadModelWithBackend requested=%{public}s status=%{public}d actual=%{public}s",
         backend.c_str(), static_cast<int>(st), g_backendDesc.c_str());
    if (st == OH_AI_STATUS_SUCCESS) {
        napi_create_int32(env, 0, &result);
        return result;
    }
    // Fall back to CPU so the app stays usable; report the original failure.
    OH_AI_ModelDestroy(&g_model);
    g_model = nullptr;
    OH_AI_ContextDestroy(&g_ctx);
    g_ctx = nullptr;
    if (!EnsureContext(4, 0)) {
        napi_create_int32(env, -1004, &result);
        return result;
    }
    g_model = OH_AI_ModelCreate();
    if (g_model != nullptr) {
        OH_AI_Status st2 = OH_AI_ModelBuild(g_model, data, len, OH_AI_MODELTYPE_MINDIR, g_ctx);
        LOGI("CPU fallback build status=%{public}d actual=%{public}s",
             static_cast<int>(st2), g_backendDesc.c_str());
    }
    napi_create_int32(env, static_cast<int32_t>(st), &result);
    return result;
}

// Build from buffer with explicit CPU tuning: thread count and operator parallelism.
static napi_value LoadModelFromBufferCfg(napi_env env, napi_callback_info info)
{
    size_t argc = 3;
    napi_value argv[3] = {nullptr, nullptr, nullptr};
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);
    void *data = nullptr;
    size_t len = 0;
    bool isAb = false;
    napi_is_arraybuffer(env, argv[0], &isAb);
    if (isAb) {
        napi_get_arraybuffer_info(env, argv[0], &data, &len);
    } else {
        bool isTa = false;
        napi_is_typedarray(env, argv[0], &isTa);
        if (isTa) {
            napi_typedarray_type type;
            size_t length = 0;
            void *td = nullptr;
            napi_value ab = nullptr;
            size_t offset = 0;
            napi_get_typedarray_info(env, argv[0], &type, &length, &td, &ab, &offset);
            data = td;
            len = length;
        }
    }
    int32_t threads = 2;
    bool parallel = false;
    if (argc >= 2) {
        napi_get_value_int32(env, argv[1], &threads);
    }
    if (argc >= 3) {
        napi_get_value_bool(env, argv[2], &parallel);
    }
    napi_value result = nullptr;
    if (data == nullptr || len == 0) {
        napi_create_int32(env, -1002, &result);
        return result;
    }
    if (g_model != nullptr) {
        OH_AI_ModelDestroy(&g_model);
        g_model = nullptr;
    }
    if (g_ctx != nullptr) {
        OH_AI_ContextDestroy(&g_ctx);
        g_ctx = nullptr;
    }
    g_ctx = OH_AI_ContextCreate();
    if (g_ctx == nullptr) {
        napi_create_int32(env, -1003, &result);
        return result;
    }
    OH_AI_DeviceInfoHandle cpu = OH_AI_DeviceInfoCreate(OH_AI_DEVICETYPE_CPU);
    if (cpu == nullptr) {
        napi_create_int32(env, -1003, &result);
        return result;
    }
    OH_AI_ContextAddDeviceInfo(g_ctx, cpu);
    if (threads > 0) {
        OH_AI_ContextSetThreadNum(g_ctx, threads);
    }
    OH_AI_ContextSetEnableParallel(g_ctx, parallel);
    g_backendDesc = "CPU (threads=" + std::to_string(threads) +
        ", parallel=" + (parallel ? "true" : "false") + ")";
    g_model = OH_AI_ModelCreate();
    if (g_model == nullptr) {
        napi_create_int32(env, -1003, &result);
        return result;
    }
    OH_AI_Status st = OH_AI_ModelBuild(g_model, data, len, OH_AI_MODELTYPE_MINDIR, g_ctx);
    LOGI("LoadModelFromBufferCfg threads=%{public}d parallel=%{public}d status=%{public}d",
         threads, parallel ? 1 : 0, static_cast<int>(st));
    napi_create_int32(env, static_cast<int32_t>(st), &result);
    return result;
}

static napi_value Predict(napi_env env, napi_callback_info info)
{
    size_t argc = 2;
    napi_value argv[2] = {nullptr, nullptr};
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);
    std::vector<int32_t> ids;
    std::vector<int32_t> mask;
    if (argc < 2 || !GetIntArray(env, argv[0], &ids) || !GetIntArray(env, argv[1], &mask)) {
        LOGE("predict: bad input arrays");
        return nullptr;
    }
    if (g_model == nullptr) {
        LOGE("predict: model not loaded");
        return nullptr;
    }

    OH_AI_TensorHandleArray inputs = OH_AI_ModelGetInputs(g_model);
    if (inputs.handle_num < 2) {
        LOGE("predict: expected 2 inputs, got %{public}zu", inputs.handle_num);
        return nullptr;
    }

    // Fill both input tensors honoring their actual dtype (converter may store int32 or int64).
    std::vector<std::vector<int64_t>> i64bufs(2);
    std::vector<std::vector<int32_t>> i32bufs(2);
    for (size_t i = 0; i < 2; i++) {
        OH_AI_TensorHandle t = inputs.handle_list[i];
        int64_t elem = OH_AI_TensorGetElementNum(t);
        OH_AI_DataType dt = OH_AI_TensorGetDataType(t);
        const std::vector<int32_t> &src = (i == 0) ? ids : mask;
        if (dt == OH_AI_DATATYPE_NUMBERTYPE_INT64) {
            i64bufs[i].resize(static_cast<size_t>(elem), 0);
            for (int64_t k = 0; k < elem && k < static_cast<int64_t>(src.size()); k++) {
                i64bufs[i][static_cast<size_t>(k)] = src[static_cast<size_t>(k)];
            }
            OH_AI_TensorSetData(t, i64bufs[i].data());
        } else {
            i32bufs[i].resize(static_cast<size_t>(elem), 0);
            for (int64_t k = 0; k < elem && k < static_cast<int64_t>(src.size()); k++) {
                i32bufs[i][static_cast<size_t>(k)] = src[static_cast<size_t>(k)];
            }
            OH_AI_TensorSetData(t, i32bufs[i].data());
        }
        LOGI("input[%{public}zu] dtype=%{public}d elems=%{public}lld", i, static_cast<int>(dt), (long long)elem);
    }

    OH_AI_TensorHandleArray outputs = OH_AI_ModelGetOutputs(g_model);
    OH_AI_Status st = OH_AI_ModelPredict(g_model, inputs, &outputs, nullptr, nullptr);
    if (st != OH_AI_STATUS_SUCCESS) {
        LOGE("predict failed status=%{public}d", static_cast<int>(st));
        return nullptr;
    }
    if (outputs.handle_num < 1) {
        LOGE("predict: no outputs");
        return nullptr;
    }
    OH_AI_TensorHandle out = outputs.handle_list[0];
    int64_t n = OH_AI_TensorGetElementNum(out);
    const float *data = static_cast<const float *>(OH_AI_TensorGetData(out));
    napi_value arr = nullptr;
    napi_create_array_with_length(env, static_cast<size_t>(n), &arr);
    for (int64_t i = 0; i < n; i++) {
        napi_value num = nullptr;
        napi_create_double(env, static_cast<double>(data[i]), &num);
        napi_set_element(env, arr, static_cast<uint32_t>(i), num);
    }
    LOGI("predict ok out_elems=%{public}lld", (long long)n);
    return arr;
}

static napi_value Init(napi_env env, napi_value exports)
{
    napi_property_descriptor desc[] = {
        {"loadModel", nullptr, LoadModel, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"loadModelFromBuffer", nullptr, LoadModelFromBuffer, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"loadModelWithBackend", nullptr, LoadModelWithBackend, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"loadModelFromBufferCfg", nullptr, LoadModelFromBufferCfg, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"diagnose", nullptr, Diagnose, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"inferenceBackend", nullptr, InferenceBackend, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"availableAccelerators", nullptr, AvailableAccelerators, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"predict", nullptr, Predict, nullptr, nullptr, nullptr, napi_default, nullptr},
    };
    napi_define_properties(env, exports, sizeof(desc) / sizeof(desc[0]), desc);
    return exports;
}

static napi_module g_module = {
    .nm_version = 1,
    .nm_flags = 0,
    .nm_filename = nullptr,
    .nm_register_func = Init,
    .nm_modname = "riskguard_native",
    .nm_priv = nullptr,
    .reserved = {0},
};

extern "C" __attribute__((constructor)) void RegisterRiskGuardNativeModule()
{
    napi_module_register(&g_module);
}
