import { AbilityConstant, UIAbility, Want } from '@kit.AbilityKit';
import { hilog } from '@kit.PerformanceAnalysisKit';
import { window } from '@kit.ArkUI';
import { runAutoTest } from '../services/AutoTest';

const TAG: string = 'RiskGuard';

export default class EntryAbility extends UIAbility {
  onCreate(want: Want, launchParam: AbilityConstant.LaunchParam): void {
    hilog.info(0x0000, TAG, 'EntryAbility onCreate');
    // Dev/CI hook: launch with `aa start ... --ps autotest npu|assess|all`
    // to run headless self-tests without touching the UI.
    const params = want.parameters;
    if (params !== undefined && params['autotest'] !== undefined) {
      const mode: string = String(params['autotest']);
      hilog.info(0x0000, TAG, 'autotest hook: mode=%{public}s', mode);
      const ctx = this.context;
      setTimeout(() => {
        runAutoTest(ctx, mode);
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
