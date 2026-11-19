import { common } from '@kit.AbilityKit';
import { hilog } from '@kit.PerformanceAnalysisKit';
import { RiskGuardService, AssessInput, AssessResult, decisionText } from '../services/RiskGuardService';
import { HistoryStore, nowText } from '../services/HistoryStore';

const TAG: string = 'RiskGuard';

interface Option {
  label: string;
  value: string;
}

const SCOPES: Option[] = [
  { label: '单个', value: 'single' },
  { label: '批量', value: 'batch' },
  { label: '递归', value: 'recursive' }
];

const AUTH_OPTIONS: Option[] = [
  { label: '已授权', value: 'approved' },
  { label: '不明确', value: 'ambiguous' },
  { label: '已拒绝', value: 'denied' }
];

const CLASS_OPTIONS: Option[] = [
  { label: '公开', value: 'public' },
  { label: '内部', value: 'internal' },
  { label: '个人', value: 'personal' },
  { label: '机密', value: 'confidential' },
  { label: '绝密', value: 'secret' }
];

const TRUST_OPTIONS: Option[] = [
  { label: '可信', value: 'trusted' },
  { label: '企业内', value: 'enterprise_approved' },
  { label: '未知外部', value: 'unknown_domain' },
  { label: '不可信外发', value: 'untrusted' }
];

function decisionColor(decision: string): string {
  if (decision === 'CONTINUE') {
    return '#1B7F3A';
  }
  if (decision === 'ASK') {
    return '#B26A00';
  }
  return '#D33A31';
}

function adviceText(decision: string): string {
  if (decision === 'CONTINUE') {
    return '可以继续，但仍受系统权限策略约束。';
  }
  if (decision === 'ASK') {
    return '先不要自动执行：向用户补充授权、确认目的地信任或恢复条件。';
  }
  return '拒绝当前操作并保留审计信息；不可恢复删除或凭据外发必须由独立策略引擎二次确认。';
}

@Component
export struct AssessView {
  @State private instruction: string = '整理下载目录';
  @State private tool: string = 'file.delete';
  @State private target: string = 'user_document';
  @State private scope: string = 'single';
  @State private countText: string = '1';
  @State private authorization: string = 'ambiguous';
  @State private classification: string = 'internal';
  @State private recoverable: boolean = false;
  @State private backupAvailable: boolean = true;
  @State private destinationTrust: string = 'trusted';

  @State private running: boolean = false;
  @State private hasResult: boolean = false;
  @State private decision: string = '';
  @State private continueScore: number = 0;
  @State private askScore: number = 0;
  @State private blockScore: number = 0;
  @State private latencyMs: number = 0;
  @State private errorText: string = '';
  @State private modelState: string = '模型准备中…';

  aboutToAppear(): void {
    this.preload();
  }

  private async preload(): Promise<void> {
    // Skip preloading during headless autotest launches (single-load processes).
    if (AppStorage.get<string>('autotestMode') !== undefined) {
      this.modelState = '自检模式';
      return;
    }
    try {
      const ctx = getContext(this) as common.UIAbilityContext;
      await RiskGuardService.get().ensureReady(ctx);
      this.modelState = '本地模型就绪 · 加载 ' + RiskGuardService.get().loadTimeMs() + 'ms';
    } catch (err) {
      this.modelState = '模型加载失败';
      this.errorText = (err as Error).message;
      hilog.error(0x0000, TAG, 'preload failed: %{public}s', this.errorText);
    }
  }

  private buildInput(): AssessInput {
    const n: number = parseInt(this.countText, 10);
    return {
      instruction: this.instruction.trim(),
      tool: this.tool.trim(),
      target: this.target.trim(),
      scope: this.scope,
      affectedCount: isNaN(n) || n < 1 ? 1 : n,
      authorization: this.authorization,
      classification: this.classification,
      recoverable: this.recoverable,
      backupAvailable: this.backupAvailable,
      destinationTrust: this.destinationTrust
    };
  }

  private async runAssess(): Promise<void> {
    if (this.instruction.trim().length === 0) {
      this.errorText = '请先填写用户指令';
      return;
    }
    if (this.tool.trim().length === 0) {
      this.errorText = '请先填写工具名称（例如 file.delete）';
      return;
    }
    this.running = true;
    this.errorText = '';
    try {
      const ctx = getContext(this) as common.UIAbilityContext;
      const result: AssessResult = await RiskGuardService.get().assess(ctx, this.buildInput());
      this.decision = result.decision;
      this.continueScore = result.continueScore;
      this.askScore = result.askScore;
      this.blockScore = result.blockScore;
      this.latencyMs = result.latencyMs;
      this.hasResult = true;
      this.modelState = '本地模型就绪 · 加载 ' + RiskGuardService.get().loadTimeMs() + 'ms';
      await HistoryStore.add(ctx, {
        time: nowText(),
        instruction: this.instruction.trim(),
        tool: this.tool.trim(),
        target: this.target.trim(),
        decision: result.decision,
        scores: '继续 ' + (result.continueScore * 100).toFixed(1) + '% / 确认 ' +
          (result.askScore * 100).toFixed(1) + '% / 阻止 ' + (result.blockScore * 100).toFixed(1) + '%'
      });
      const scoreText: string = (result.continueScore * 100).toFixed(1) + '/' +
        (result.askScore * 100).toFixed(1) + '/' + (result.blockScore * 100).toFixed(1);
      hilog.info(0x0000, TAG, 'ASSESS ok decision=%{public}s scores=%{public}s latency=%{public}dms',
        result.decision, scoreText, result.latencyMs);
    } catch (err) {
      this.errorText = (err as Error).message;
      hilog.error(0x0000, TAG, 'ASSESS failed: %{public}s', this.errorText);
    }
    this.running = false;
  }

  build() {
    Scroll() {
      Column() {
        // Header
        Column({ space: 4 }) {
          Text('风险守卫')
            .fontSize(28)
            .fontWeight(FontWeight.Bold)
            .fontColor('#182431')
          Text('Agent 操作风险评估 · 本地离线 · 数据不出设备')
            .fontSize(13)
            .fontColor('#6B7785')
        }
        .alignItems(HorizontalAlign.Start)
        .width('100%')
        .margin({ bottom: 16 })

        Row({ space: 16 }) {
          // Input card (left)
          Column({ space: 12 }) {
            Text('操作信息').fontSize(16).fontWeight(FontWeight.Medium).fontColor('#182431')

          Column({ space: 6 }) {
            Text('用户指令').fontSize(13).fontColor('#6B7785')
            TextArea({ text: this.instruction, placeholder: '例如：整理下载目录 / 永久删除用户重要资料' })
              .height(76)
              .fontSize(14)
              .backgroundColor('#F7F8FA')
              .onChange((v: string) => { this.instruction = v; })
          }.alignItems(HorizontalAlign.Start).width('100%')

          Row({ space: 12 }) {
            Column({ space: 6 }) {
              Text('工具').fontSize(13).fontColor('#6B7785')
              TextInput({ text: this.tool, placeholder: '如 file.delete' })
                .fontSize(14)
                .backgroundColor('#F7F8FA')
                .onChange((v: string) => { this.tool = v; })
            }.layoutWeight(1).alignItems(HorizontalAlign.Start)

            Column({ space: 6 }) {
              Text('操作目标').fontSize(13).fontColor('#6B7785')
              TextInput({ text: this.target, placeholder: '如 user_document' })
                .fontSize(14)
                .backgroundColor('#F7F8FA')
                .onChange((v: string) => { this.target = v; })
            }.layoutWeight(1).alignItems(HorizontalAlign.Start)
          }.width('100%')

          Column({ space: 6 }) {
            Text('操作范围').fontSize(13).fontColor('#6B7785')
            Row({ space: 8 }) {
              ForEach(SCOPES, (opt: Option) => {
                Text(opt.label)
                  .fontSize(13)
                  .fontColor(this.scope === opt.value ? '#0A59F7' : '#4A5568')
                  .backgroundColor(this.scope === opt.value ? '#E8F0FE' : '#F1F3F5')
                  .borderRadius(8)
                  .border({ width: 1, color: this.scope === opt.value ? '#0A59F7' : '#E3E6EA' })
                  .padding({ left: 14, right: 14, top: 6, bottom: 6 })
                  .onClick(() => { this.scope = opt.value; })
              }, (opt: Option) => opt.value)

              if (this.scope !== 'single') {
                Text('数量').fontSize(13).fontColor('#6B7785').margin({ left: 12 })
                TextInput({ text: this.countText })
                  .width(80)
                  .fontSize(14)
                  .type(InputType.Number)
                  .backgroundColor('#F7F8FA')
                  .onChange((v: string) => { this.countText = v; })
              }
            }.width('100%')
          }.alignItems(HorizontalAlign.Start).width('100%')

          Column({ space: 6 }) {
            Text('授权状态').fontSize(13).fontColor('#6B7785')
            Row({ space: 8 }) {
              ForEach(AUTH_OPTIONS, (opt: Option) => {
                Text(opt.label)
                  .fontSize(13)
                  .fontColor(this.authorization === opt.value ? '#0A59F7' : '#4A5568')
                  .backgroundColor(this.authorization === opt.value ? '#E8F0FE' : '#F1F3F5')
                  .borderRadius(8)
                  .border({ width: 1, color: this.authorization === opt.value ? '#0A59F7' : '#E3E6EA' })
                  .padding({ left: 14, right: 14, top: 6, bottom: 6 })
                  .onClick(() => { this.authorization = opt.value; })
              }, (opt: Option) => opt.value)
            }.width('100%')
          }.alignItems(HorizontalAlign.Start).width('100%')

          Column({ space: 6 }) {
            Text('数据分级').fontSize(13).fontColor('#6B7785')
            Row({ space: 8 }) {
              ForEach(CLASS_OPTIONS, (opt: Option) => {
                Text(opt.label)
                  .fontSize(13)
                  .fontColor(this.classification === opt.value ? '#0A59F7' : '#4A5568')
                  .backgroundColor(this.classification === opt.value ? '#E8F0FE' : '#F1F3F5')
                  .borderRadius(8)
                  .border({ width: 1, color: this.classification === opt.value ? '#0A59F7' : '#E3E6EA' })
                  .padding({ left: 14, right: 14, top: 6, bottom: 6 })
                  .onClick(() => { this.classification = opt.value; })
              }, (opt: Option) => opt.value)
            }.width('100%')
          }.alignItems(HorizontalAlign.Start).width('100%')

          Column({ space: 6 }) {
            Text('目的地信任').fontSize(13).fontColor('#6B7785')
            Row({ space: 8 }) {
              ForEach(TRUST_OPTIONS, (opt: Option) => {
                Text(opt.label)
                  .fontSize(13)
                  .fontColor(this.destinationTrust === opt.value ? '#0A59F7' : '#4A5568')
                  .backgroundColor(this.destinationTrust === opt.value ? '#E8F0FE' : '#F1F3F5')
                  .borderRadius(8)
                  .border({ width: 1, color: this.destinationTrust === opt.value ? '#0A59F7' : '#E3E6EA' })
                  .padding({ left: 14, right: 14, top: 6, bottom: 6 })
                  .onClick(() => { this.destinationTrust = opt.value; })
              }, (opt: Option) => opt.value)
            }.width('100%')
          }.alignItems(HorizontalAlign.Start).width('100%')

          Row({ space: 28 }) {
            Row({ space: 8 }) {
              Toggle({ type: ToggleType.Switch, isOn: this.recoverable })
                .onChange((v: boolean) => { this.recoverable = v; })
              Text('操作可恢复').fontSize(13).fontColor('#4A5568')
            }
            Row({ space: 8 }) {
              Toggle({ type: ToggleType.Switch, isOn: this.backupAvailable })
                .onChange((v: boolean) => { this.backupAvailable = v; })
              Text('存在备份').fontSize(13).fontColor('#4A5568')
            }
          }.width('100%').margin({ top: 2 })

          if (this.errorText.length > 0) {
            Text(this.errorText).fontSize(12).fontColor('#D33A31').width('100%')
          }

          Row({ space: 12 }) {
            Button(this.running ? '评估中…' : '开始评估')
              .enabled(!this.running)
              .fontSize(15)
              .height(42)
              .width(160)
              .backgroundColor('#0A59F7')
              .onClick(() => { this.runAssess(); })
            Text(this.modelState).fontSize(12).fontColor('#6B7785')
          }.width('100%').margin({ top: 4 })
        }
          .alignItems(HorizontalAlign.Start)
          .padding(22)
          .backgroundColor('#FFFFFF')
          .borderRadius(16)
          .layoutWeight(1)

          // Result panel (right)
          Column() {
            if (this.hasResult) {
              Column({ space: 14 }) {
            Row({ space: 12 }) {
              Text(decisionText(this.decision))
                .fontSize(20)
                .fontWeight(FontWeight.Bold)
                .fontColor('#FFFFFF')
                .backgroundColor(decisionColor(this.decision))
                .borderRadius(10)
                .padding({ left: 16, right: 16, top: 6, bottom: 6 })
              Text(this.decision)
                .fontSize(14)
                .fontColor(decisionColor(this.decision))
              Blank()
              Text(this.latencyMs + 'ms').fontSize(12).fontColor('#6B7785')
            }.width('100%')

            Column({ space: 8 }) {
              Row({ space: 10 }) {
                Text('可继续').fontSize(13).width(64).fontColor('#4A5568')
                Progress({ value: this.continueScore * 100, total: 100, type: ProgressType.Linear })
                  .color('#1B7F3A').layoutWeight(1)
                Text((this.continueScore * 100).toFixed(1) + '%').fontSize(12).width(64)
                  .textAlign(TextAlign.End).fontColor('#4A5568')
              }.width('100%')
              Row({ space: 10 }) {
                Text('需确认').fontSize(13).width(64).fontColor('#4A5568')
                Progress({ value: this.askScore * 100, total: 100, type: ProgressType.Linear })
                  .color('#B26A00').layoutWeight(1)
                Text((this.askScore * 100).toFixed(1) + '%').fontSize(12).width(64)
                  .textAlign(TextAlign.End).fontColor('#4A5568')
              }.width('100%')
              Row({ space: 10 }) {
                Text('应阻止').fontSize(13).width(64).fontColor('#4A5568')
                Progress({ value: this.blockScore * 100, total: 100, type: ProgressType.Linear })
                  .color('#D33A31').layoutWeight(1)
                Text((this.blockScore * 100).toFixed(1) + '%').fontSize(12).width(64)
                  .textAlign(TextAlign.End).fontColor('#4A5568')
              }.width('100%')
            }.width('100%')

            Text(adviceText(this.decision))
              .fontSize(13)
              .fontColor('#182431')
              .width('100%')
            Text('置信度为模型原始输出；模型在合成数据上高度拟合，数值偏饱和（常见 99%+），请结合策略判断。')
              .fontSize(11)
              .fontColor('#8A94A6')
              .width('100%')
            Text('本结果只是风险建议，不构成授权；高危操作必须由确定性策略引擎兜底。')
              .fontSize(11)
              .fontColor('#999999')
              .width('100%')
              }
              .alignItems(HorizontalAlign.Start)
              .padding(22)
              .backgroundColor('#FFFFFF')
              .borderRadius(16)
              .width('100%')
            } else {
              Column({ space: 10 }) {
                Text('等待评估').fontSize(16).fontWeight(FontWeight.Medium).fontColor('#182431')
                Text('填写左侧操作信息后，点击“开始评估”。').fontSize(13).fontColor('#6B7785')
                Divider().margin({ top: 8, bottom: 8 })
                Text('决策说明').fontSize(13).fontWeight(FontWeight.Medium).fontColor('#182431')
                Row({ space: 8 }) {
                  Text('可继续').fontSize(12).fontColor('#FFFFFF').backgroundColor('#1B7F3A')
                    .borderRadius(6).padding({ left: 8, right: 8, top: 3, bottom: 3 })
                  Text('已授权、低风险、可恢复').fontSize(12).fontColor('#6B7785')
                }
                Row({ space: 8 }) {
                  Text('需确认').fontSize(12).fontColor('#FFFFFF').backgroundColor('#B26A00')
                    .borderRadius(6).padding({ left: 8, right: 8, top: 3, bottom: 3 })
                  Text('授权不明确或目的地待核实').fontSize(12).fontColor('#6B7785')
                }
                Row({ space: 8 }) {
                  Text('应阻止').fontSize(12).fontColor('#FFFFFF').backgroundColor('#D33A31')
                    .borderRadius(6).padding({ left: 8, right: 8, top: 3, bottom: 3 })
                  Text('明显越权、不可恢复或敏感外发').fontSize(12).fontColor('#6B7785')
                }
                Text('模型基于合成数据训练，输出仅为风险建议，不构成授权。')
                  .fontSize(11).fontColor('#999999').margin({ top: 6 })
              }
              .alignItems(HorizontalAlign.Start)
              .padding(22)
              .backgroundColor('#FFFFFF')
              .borderRadius(16)
              .width('100%')
            }
          }
          .width('44%')
        }
        .width('100%')
        .alignItems(VerticalAlign.Top)
      }
      .constraintSize({ maxWidth: 920 })
      .padding({ left: 28, right: 28, top: 26, bottom: 30 })
      .width('100%')
    }
    .width('100%')
    .height('100%')
    .backgroundColor('#F1F3F5')
  }
}
