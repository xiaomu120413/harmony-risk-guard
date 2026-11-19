import { AbilityConstant, UIAbility, Want } from '@kit.AbilityKit';
import { hilog } from '@kit.PerformanceAnalysisKit';
import { window } from '@kit.ArkUI';
import { runAutoTest } from '../services/AutoTest';

const TAG: string = 'RiskGuard';

export default class EntryAbility extends UIAbility {
  onCreate(want: Want, launchParam: AbilityConstant.LaunchParam): void {
    hilog.info(0x0000, TAG, 'EntryAbility onCreate');
    // Dev/CI hook: launch with
    //   aa start ... --ps autotest npu|assess|all|perf [--pi threads 4] [--pb parallel true]
    // to run headless self-tests without touching the UI.
    const params = want.parameters;
    if (params !== undefined && params['autotest'] !== undefined) {
      const mode: string = String(params['autotest']);
      let threads: number = 0;
      let parallel: boolean = false;
      if (params['threads'] !== undefined) {
        const n: number = parseInt(String(params['threads']), 10);
        threads = isNaN(n) ? 0 : n;
      }
      if (params['parallel'] !== undefined) {
        parallel = String(params['parallel']) === 'true' || String(params['parallel']) === '1';
      }
      // Suppress the UI's own model preload so perf runs measure a single load.
      AppStorage.setOrCreate<string>('autotestMode', mode);
      hilog.info(0x0000, TAG, 'autotest hook: mode=%{public}s threads=%{public}d parallel=%{public}d',
        mode, threads, parallel ? 1 : 0);
      const ctx = this.context;
      setTimeout(() => {
        runAutoTest(ctx, mode, threads, parallel);
      }, 800);
    }
  }

  onWindowStageCreate(windowStage: window.WindowStage): void {
    windowStage.loadContent('pages/Index', (err) => {
      if (err.code) {
        hilog.error(0x0000, TAG, 'loadContent failed: %{public}s', JSON.stringify(err));
        return;
      }
      hilog.info(0x0000, TAG, 'loadContent succeeded');
    });
  }

  onDestroy(): void {
    hilog.info(0x0000, TAG, 'EntryAbility onDestroy');
  }
}
