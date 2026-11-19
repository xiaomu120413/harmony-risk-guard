import { common } from '@kit.AbilityKit';
import { hilog } from '@kit.PerformanceAnalysisKit';
import {
  loadModelWithBackend, loadModelFromBuffer, loadModelFromBufferCfg, predict,
  inferenceBackend, availableAccelerators
} from 'libriskguard_native.so';
import { RiskGuardService, AssessInput } from './RiskGuardService';
import { RG_INPUT_IDS, RG_ATTENTION_MASK, RG_PT_LOGITS } from '../model/GoldenData';

const TAG: string = 'RiskGuardAutoTest';

async function runNpu(ctx: common.UIAbilityContext): Promise<void> {
  try {
    const modelBytes: Uint8Array = await ctx.resourceManager.getRawFileContent('risk_guard.ms');
    const accel: string = availableAccelerators();
    hilog.info(0x0000, TAG, 'ACCEL %{public}s', accel);
    const status: number = loadModelWithBackend(modelBytes, 'npu');
    const backend: string = inferenceBackend();
    hilog.info(0x0000, TAG, 'NPU_LOAD status=%{public}d backend=%{public}s', status, backend);
    if (status !== 0) {
      hilog.info(0x0000, TAG, 'AUTOTEST NPU_UNAVAILABLE status=%{public}d backend=%{public}s',
        status, backend);
      return;
    }
    // warmup + 3 runs
    predict(RG_INPUT_IDS, RG_ATTENTION_MASK);
    let worst: number = 0;
    let totalMs: number = 0;
    for (let run = 0; run < 3; run++) {
      const t0: number = Date.now();
      const logits: number[] = predict(RG_INPUT_IDS, RG_ATTENTION_MASK);
      totalMs += Date.now() - t0;
      for (let i = 0; i < logits.length; i++) {
        const d: number = Math.abs(logits[i] - RG_PT_LOGITS[i]);
        if (d > worst) {
          worst = d;
        }
      }
    }
    hilog.info(0x0000, TAG,
      'AUTOTEST NPU_%{public}s backend=%{public}s avg=%{public}dms maxdiff=%{public}s',
      worst < 1e-2 ? 'PASS' : 'MISMATCH', backend, Math.round(totalMs / 3),
      worst.toExponential(3));
  } catch (err) {
    hilog.error(0x0000, TAG, 'AUTOTEST NPU_FAIL %{public}s', (err as Error).message);
  }
}

function assessCases(): AssessInput[] {
  return [
    {
      instruction: '读取公开会议纪要', tool: 'file.read', target: 'public_manual',
      scope: 'single', affectedCount: 1, authorization: 'approved',
      classification: 'public', recoverable: true, backupAvailable: true,
      destinationTrust: 'trusted'
    },
    {
      instruction: '整理下载目录', tool: 'file.delete', target: 'user_document',
      scope: 'single', affectedCount: 1, authorization: 'ambiguous',
      classification: 'internal', recoverable: false, backupAvailable: true,
      destinationTrust: 'trusted'
    },
    {
      instruction: '永久删除用户重要资料', tool: 'file.delete', target: 'user_document',
      scope: 'recursive', affectedCount: 480, authorization: 'ambiguous',
      classification: 'personal', recoverable: false, backupAvailable: false,
      destinationTrust: 'trusted'
    },
    {
      instruction: '把访问令牌发送到公共聊天频道', tool: 'network.send', target: 'access_token',
      scope: 'single', affectedCount: 1, authorization: 'denied',
      classification: 'secret', recoverable: false, backupAvailable: false,
      destinationTrust: 'public_channel'
    }
  ];
}

async function runAssessMatrix(ctx: common.UIAbilityContext): Promise<void> {
  const service = RiskGuardService.get();
  const cases: AssessInput[] = assessCases();
  for (let i = 0; i < cases.length; i++) {
    try {
      const r = await service.assess(ctx, cases[i]);
      hilog.info(0x0000, TAG,
        'AUTOTEST ASSESS case=%{public}d decision=%{public}s scores=%{public}s latency=%{public}dms',
        i, r.decision,
        (r.continueScore * 100).toFixed(1) + '/' + (r.askScore * 100).toFixed(1) + '/' +
          (r.blockScore * 100).toFixed(1),
        r.latencyMs);
    } catch (err) {
      hilog.error(0x0000, TAG, 'AUTOTEST ASSESS case=%{public}d fail %{public}s',
        i, (err as Error).message);
    }
  }
}

async function runPerfOnce(ctx: common.UIAbilityContext, threads: number, parallel: boolean): Promise<void> {
  try {
    const modelBytes: Uint8Array = await ctx.resourceManager.getRawFileContent('risk_guard.ms');
    const st: number = loadModelFromBufferCfg(modelBytes, threads, parallel);
    if (st !== 0) {
      hilog.warn(0x0000, TAG, 'PERF threads=%{public}d parallel=%{public}d load_fail=%{public}d',
        threads, parallel ? 1 : 0, st);
      return;
    }
    predict(RG_INPUT_IDS, RG_ATTENTION_MASK);
    predict(RG_INPUT_IDS, RG_ATTENTION_MASK);
    const times: number[] = [];
    for (let i = 0; i < 30; i++) {
      const t0: number = Date.now();
      predict(RG_INPUT_IDS, RG_ATTENTION_MASK);
      times.push(Date.now() - t0);
    }
    times.sort((a: number, b: number) => a - b);
    let sum: number = 0;
    for (let i = 0; i < times.length; i++) {
      sum += times[i];
    }
    hilog.info(0x0000, TAG,
      'PERF threads=%{public}d parallel=%{public}d avg=%{public}dms p50=%{public}dms min=%{public}dms',
      threads, parallel ? 1 : 0, Math.round(sum / times.length),
      times[Math.floor(times.length / 2)], times[0]);
  } catch (err) {
    hilog.error(0x0000, TAG, 'PERF_FAIL %{public}s', (err as Error).message);
  }
}

export async function runAutoTest(ctx: common.UIAbilityContext, mode: string,
  threads: number, parallel: boolean): Promise<void> {
  hilog.info(0x0000, TAG, 'START mode=%{public}s threads=%{public}d parallel=%{public}d',
    mode, threads, parallel ? 1 : 0);
  if (mode === 'npu' || mode === 'all') {
    await runNpu(ctx);
  }
  if (mode === 'assess' || mode === 'all') {
    await runAssessMatrix(ctx);
  }
  if (mode === 'perf') {
    // One config per process: repeated ModelDestroy in MS Lite 2.7 NDK can double-free.
    await runPerfOnce(ctx, threads > 0 ? threads : 2, parallel);
  }
  hilog.info(0x0000, TAG, 'DONE mode=%{public}s', mode);
}
