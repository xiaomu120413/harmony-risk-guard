import { mindSporeLite } from '@kit.MindSporeLiteKit';
import { hilog } from '@kit.PerformanceAnalysisKit';
import { fileIo } from '@kit.CoreFileKit';
import { common } from '@kit.AbilityKit';
import { router } from '@kit.ArkUI';
import { loadModelFromBuffer, predict } from 'libriskguard_native.so';
import {
  RG_SAMPLE_ID, RG_INPUT_IDS, RG_ATTENTION_MASK, RG_PT_LOGITS, RG_PT_LABEL, RG_LABELS
} from '../model/GoldenData';
import { TOKENIZER_CASES } from '../model/TokenizerGolden';
import { BertTokenizer } from '../model/BertTokenizer';

const TAG: string = 'RiskGuardDiag';
const MLP_GOLDEN: number[] = [0.161677, -0.013502, -0.003518];
const MLP_SAMPLE: number[] = [0.1, 0.2, 0.3, 0.4];

@Entry
@Component
struct Diagnostics {
  @State private running: boolean = false;
  @State private mlpStatus: string = '未运行';
  @State private mlpDetail: string = '';
  @State private rgStatus: string = '未运行';
  @State private rgDetail: string = '';
  @State private tokStatus: string = '未运行';
  @State private tokDetail: string = '';
  @State private ndStatus: string = '未运行';
  @State private ndDetail: string = '';
  @State private benchStatus: string = '未运行';
  @State private benchDetail: string = '';
  @State private benchRunning: boolean = false;

  private async copyRaw(ctx: common.UIAbilityContext, name: string): Promise<string> {
    const raw: Uint8Array = await ctx.resourceManager.getRawFileContent(name);
    const path: string = ctx.filesDir + '/' + name;
    const file = fileIo.openSync(path,
      fileIo.OpenMode.CREATE | fileIo.OpenMode.READ_WRITE | fileIo.OpenMode.TRUNC);
    fileIo.writeSync(file.fd, raw.buffer);
    fileIo.closeSync(file);
    return path;
  }

  private async runMlpChecks(ctx: common.UIAbilityContext): Promise<void> {
    try {
      const path: string = await this.copyRaw(ctx, 'probe_mlp.ms');
      const model = await mindSporeLite.loadModelFromFile(path);
      const inputs = model.getInputs();
      inputs[0].setData(new Float32Array(MLP_SAMPLE).buffer);
      const outputs = await model.predict(inputs);
      const values = new Float32Array(outputs[0].getData());
      let worst: number = 0;
      for (let i = 0; i < values.length; i++) {
        const d = Math.abs(values[i] - MLP_GOLDEN[i]);
        if (d > worst) {
          worst = d;
        }
      }
      this.mlpStatus = worst < 1e-3 ? 'M2-A PASS' : 'M2-A MISMATCH';
      this.mlpDetail = 'max|Δ|=' + worst.toExponential(3) + ' (MindSporeLiteKit / MLP)';
      hilog.info(0x0000, TAG, 'MLP %{public}s', this.mlpDetail);
    } catch (err) {
      this.mlpStatus = 'M2-A FAILED';
      this.mlpDetail = (err as Error).message;
    }
  }

  private async runNativeChecks(ctx: common.UIAbilityContext): Promise<void> {
    try {
      const modelBytes: Uint8Array = await ctx.resourceManager.getRawFileContent('risk_guard.ms');
      const status: number = loadModelFromBuffer(modelBytes);
      if (status !== 0) {
        this.ndStatus = 'M2-D FAILED';
        this.ndDetail = 'load status=' + status;
        return;
      }
      const t0: number = Date.now();
      const logits: number[] = predict(RG_INPUT_IDS, RG_ATTENTION_MASK);
      const inferMs: number = Date.now() - t0;
      let worst: number = 0;
      for (let i = 0; i < logits.length; i++) {
        const d = Math.abs(logits[i] - RG_PT_LOGITS[i]);
        if (d > worst) {
          worst = d;
        }
      }
      this.ndStatus = worst < 1e-2 ? 'M2-D PASS' : 'M2-D MISMATCH';
      this.ndDetail = 'max|Δ|=' + worst.toExponential(3) + ' infer=' + inferMs + 'ms (NAPI + C API)';
      hilog.info(0x0000, TAG, 'ND %{public}s', this.ndDetail);
    } catch (err) {
      this.ndStatus = 'M2-D FAILED';
      this.ndDetail = (err as Error).message;
    }
  }

  private async runGoldenAndTokenizer(ctx: common.UIAbilityContext): Promise<void> {
    try {
      const vocabBytes: Uint8Array = await ctx.resourceManager.getRawFileContent('vocab.txt');
      const tok = BertTokenizer.load(vocabBytes);
      // M2-C: tokenizer parity
      let matched: number = 0;
      for (let c = 0; c < TOKENIZER_CASES.length; c++) {
        const item = TOKENIZER_CASES[c];
        const ids: number[] = tok.encode(item.text);
        let same: boolean = ids.length === item.ids.length;
        if (same) {
          for (let i = 0; i < ids.length; i++) {
            if (ids[i] !== item.ids[i]) {
              same = false;
              break;
            }
          }
        }
        if (same) {
          matched++;
        }
      }
      this.tokStatus = matched === TOKENIZER_CASES.length ? 'M2-C PASS' : 'M2-C MISMATCH';
      this.tokDetail = matched + '/' + TOKENIZER_CASES.length + ' cases match';
      // M2-B via MindSporeLiteKit
      const path: string = await this.copyRaw(ctx, 'risk_guard.ms');
      const model = await mindSporeLite.loadModelFromFile(path);
      const inputs = model.getInputs();
      inputs[0].setData(new Int32Array(RG_INPUT_IDS).buffer);
      inputs[1].setData(new Int32Array(RG_ATTENTION_MASK).buffer);
      const outputs = await model.predict(inputs);
      const values = new Float32Array(outputs[0].getData());
      let worst: number = 0;
      for (let i = 0; i < values.length; i++) {
        const d = Math.abs(values[i] - RG_PT_LOGITS[i]);
        if (d > worst) {
          worst = d;
        }
      }
      this.rgStatus = worst < 1e-2 ? 'M2-B PASS' : 'M2-B MISMATCH';
      this.rgDetail = 'max|Δ|=' + worst.toExponential(3) + ' (MindSporeLiteKit / MiniRBT)';
      hilog.info(0x0000, TAG, 'RG %{public}s TOK %{public}s', this.rgDetail, this.tokDetail);
    } catch (err) {
      this.rgStatus = 'M2-B FAILED';
      this.rgDetail = (err as Error).message;
    }
  }

  private async runAll(): Promise<void> {
    this.running = true;
    this.mlpStatus = '运行中…';
    this.rgStatus = '运行中…';
    this.tokStatus = '运行中…';
    this.ndStatus = '运行中…';
    const ctx = getContext(this) as common.UIAbilityContext;
    await this.runMlpChecks(ctx);
    await this.runGoldenAndTokenizer(ctx);
    await this.runNativeChecks(ctx);
    this.running = false;
  }

  private finishBench(times: number[]): void {
    times.sort((a: number, b: number) => a - b);
    let sum: number = 0;
    for (let i = 0; i < times.length; i++) {
      sum += times[i];
    }
    const mean: number = sum / times.length;
    const p50: number = times[Math.floor(times.length * 0.5)];
    const p95: number = times[Math.floor(times.length * 0.95)];
    this.benchStatus = 'M2-F DONE';
    this.benchDetail = 'n=' + times.length + ' mean=' + mean.toFixed(1) + 'ms p50=' + p50 +
      'ms p95=' + p95 + 'ms min=' + times[0] + 'ms max=' + times[times.length - 1] + 'ms';
    hilog.info(0x0000, TAG, 'BENCH %{public}s', this.benchDetail);
    this.benchRunning = false;
  }

  private benchChunk(times: number[], total: number): void {
    const end: number = Math.min(times.length + 25, total);
    for (let i = times.length; i < end; i++) {
      const t0: number = Date.now();
      predict(RG_INPUT_IDS, RG_ATTENTION_MASK);
      times.push(Date.now() - t0);
    }
    if (times.length < total) {
      setTimeout(() => {
        this.benchChunk(times, total);
      }, 0);
    } else {
      this.finishBench(times);
    }
  }

  private runBenchmark(): void {
    this.benchRunning = true;
    this.benchStatus = '运行中…';
    this.benchDetail = '-';
    predict(RG_INPUT_IDS, RG_ATTENTION_MASK);
    this.benchChunk([], 1000);
  }

  @Builder
  checkRow(title: string, status: string, detail: string) {
    Column({ space: 4 }) {
      Row() {
        Text(title).fontSize(14).fontWeight(FontWeight.Medium).fontColor('#182431')
        Blank()
        Text(status).fontSize(13)
          .fontColor(status.indexOf('PASS') >= 0 ? '#1B7F3A'
            : (status.indexOf('FAILED') >= 0 || status.indexOf('MISMATCH') >= 0 ? '#D33A31' : '#6B7785'))
      }.width('100%')
      Text(detail).fontSize(11).fontColor('#8A94A6').width('100%')
    }
    .alignItems(HorizontalAlign.Start)
    .padding(14).backgroundColor('#FFFFFF').borderRadius(12).width('100%')
  }

  build() {
    Scroll() {
      Column({ space: 12 }) {
        Row({ space: 12 }) {
          Button('‹ 返回')
            .fontSize(13).height(32).backgroundColor('#FFFFFF').fontColor('#0A59F7')
            .border({ width: 1, color: '#BFD3F2' })
            .onClick(() => { router.back(); })
          Column({ space: 2 }) {
            Text('自检与诊断').fontSize(22).fontWeight(FontWeight.Bold).fontColor('#182431')
            Text('开发验证页 · 普通使用无需进入').fontSize(12).fontColor('#6B7785')
          }.alignItems(HorizontalAlign.Start)
        }.width('100%')

        Button(this.running ? '自检运行中…' : '运行全部自检')
          .enabled(!this.running)
          .height(40).width(180).backgroundColor('#0A59F7')
          .onClick(() => { this.runAll(); })

        this.checkRow('M2-A 转换器与运行时（MLP）', this.mlpStatus, this.mlpDetail)
        this.checkRow('M2-B 真实模型（MiniRBT）', this.rgStatus, this.rgDetail)
        this.checkRow('M2-C 分词一致性（31 例）', this.tokStatus, this.tokDetail)
        this.checkRow('M2-D 原生 NAPI（C API）', this.ndStatus, this.ndDetail)

        Column({ space: 8 }) {
          Row({ space: 12 }) {
            Button(this.benchRunning ? '基准运行中…' : '运行 1000 次基准')
              .enabled(!this.benchRunning)
              .height(34).fontSize(13)
              .backgroundColor('#FFFFFF').fontColor('#0A59F7')
              .border({ width: 1, color: '#BFD3F2' })
              .onClick(() => { this.runBenchmark(); })
            Text(this.benchStatus).fontSize(13).fontColor('#4A5568')
          }.width('100%')
          Text(this.benchDetail).fontSize(11).fontColor('#8A94A6').width('100%')
        }
        .alignItems(HorizontalAlign.Start)
        .padding(14).backgroundColor('#FFFFFF').borderRadius(12).width('100%')

        Text('MLP 与 MiniRBT 的基准样例来自 Windows 侧 PyTorch 参考输出（tests/golden_cases.jsonl）。')
          .fontSize(11).fontColor('#999999').width('100%')
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
