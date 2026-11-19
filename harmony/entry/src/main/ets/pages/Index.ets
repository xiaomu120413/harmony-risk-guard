import { mindSporeLite } from '@kit.MindSporeLiteKit';
import { hilog } from '@kit.PerformanceAnalysisKit';
import { fileIo } from '@kit.CoreFileKit';
import { common } from '@kit.AbilityKit';
import {
  RG_SAMPLE_ID, RG_INPUT_IDS, RG_ATTENTION_MASK, RG_PT_LOGITS, RG_PT_LABEL, RG_LABELS
} from '../model/GoldenData';
import { TOKENIZER_CASES } from '../model/TokenizerGolden';
import { BertTokenizer } from '../model/BertTokenizer';
import { loadModel, loadModelFromBuffer, diagnose, predict } from 'libriskguard_native.so';

const TAG: string = 'RiskGuardProbe';
// M2-A probe: PyTorch CPU reference for the 4-16-3 MLP (seed=42)
const MLP_GOLDEN: number[] = [0.161677, -0.013502, -0.003518];
const MLP_SAMPLE: number[] = [0.1, 0.2, 0.3, 0.4];

function softmax(v: number[]): number[] {
  let m: number = -1e30;
  for (let i = 0; i < v.length; i++) {
    if (v[i] > m) {
      m = v[i];
    }
  }
  let sum: number = 0;
  const e: number[] = [];
  for (let i = 0; i < v.length; i++) {
    const x: number = Math.exp(v[i] - m);
    e.push(x);
    sum += x;
  }
  const out: number[] = [];
  for (let i = 0; i < e.length; i++) {
    out.push(e[i] / sum);
  }
  return out;
}

@Entry
@Component
struct Index {
  @State private mlpStatus: string = 'pending';
  @State private mlpOk: boolean = false;
  @State private mlpLogits: string = '-';
  @State private mlpDiff: string = '-';
  @State private rgStatus: string = 'pending';
  @State private rgOk: boolean = false;
  @State private rgLogits: string = '-';
  @State private rgDiff: string = '-';
  @State private rgLabel: string = '-';
  @State private rgDetail: string = '';
  @State private tokStatus: string = 'pending';
  @State private tokOk: boolean = false;
  @State private tokDetail: string = '';
  @State private ndStatus: string = 'pending';
  @State private ndOk: boolean = false;
  @State private ndLogits: string = '-';
  @State private ndDiff: string = '-';
  @State private ndDetail: string = '';
  // M2-E sandbox
  @State private sbInstruction: string = '整理下载目录';
  @State private sbTool: string = 'file.delete';
  @State private sbTarget: string = 'user_document';
  @State private sbAuth: string = 'ambiguous';
  @State private sbRecoverable: boolean = false;
  @State private sbTrust: string = 'trusted';
  @State private sbScores: string = '-';
  @State private sbDecision: string = '-';
  @State private sbPolicy: string = '';
  @State private sbRunning: boolean = false;
  // M2-F benchmark
  @State private benchStatus: string = 'idle';
  @State private benchDetail: string = '-';
  @State private benchRunning: boolean = false;
  private tokenizer: BertTokenizer | null = null;

  aboutToAppear(): void {
    this.runAll();
  }

  private async copyRawToSandbox(ctx: common.UIAbilityContext, name: string): Promise<string> {
    const raw: Uint8Array = await ctx.resourceManager.getRawFileContent(name);
    const path: string = ctx.filesDir + '/' + name;
    const file = fileIo.openSync(path,
      fileIo.OpenMode.CREATE | fileIo.OpenMode.READ_WRITE | fileIo.OpenMode.TRUNC);
    fileIo.writeSync(file.fd, raw.buffer);
    fileIo.closeSync(file);
    return path;
  }

  // Fill an integer tensor from token ids, honoring the tensor's actual dtype.
  private fillIntTensor(t: mindSporeLite.MSTensor, ids: number[]): number {
    const is64: boolean = t.dtype === mindSporeLite.DataType.NUMBER_TYPE_INT64;
    const width: number = is64 ? 8 : 4;
    const buf = new ArrayBuffer(ids.length * width);
    const view = new DataView(buf);
    for (let i = 0; i < ids.length; i++) {
      if (is64) {
        view.setInt32(i * 8, ids[i], true);      // low 32 bits (ids < 2^31)
        view.setInt32(i * 8 + 4, 0, true);       // high 32 bits
      } else {
        view.setInt32(i * 4, ids[i], true);
      }
    }
    t.setData(buf);
    return width;
  }

  private async runMlp(ctx: common.UIAbilityContext): Promise<void> {
    try {
      const path: string = await this.copyRawToSandbox(ctx, 'probe_mlp.ms');
      const model = await mindSporeLite.loadModelFromFile(path);
      const inputs = model.getInputs();
      const x = new Float32Array(MLP_SAMPLE);
      inputs[0].setData(x.buffer);
      const outputs = await model.predict(inputs);
      const values = new Float32Array(outputs[0].getData());
      let text: string = '';
      let worst: number = 0;
      for (let i = 0; i < values.length; i++) {
        text += (i > 0 ? ', ' : '') + values[i].toFixed(6);
        const d = Math.abs(values[i] - MLP_GOLDEN[i]);
        if (d > worst) {
          worst = d;
        }
      }
      this.mlpLogits = text;
      this.mlpDiff = worst.toExponential(3);
      this.mlpOk = worst < 1e-3;
      this.mlpStatus = this.mlpOk ? 'M2-A PASS' : 'M2-A MISMATCH';
      hilog.info(0x0000, TAG, 'MLP_OK logits=[%{public}s] maxdiff=%{public}s',
        text, this.mlpDiff);
    } catch (err) {
      this.mlpOk = false;
      this.mlpStatus = 'M2-A FAILED';
      this.mlpLogits = (err as Error).message;
      hilog.error(0x0000, TAG, 'MLP_FAIL %{public}s', (err as Error).message);
    }
  }

  private async runRiskGuard(ctx: common.UIAbilityContext): Promise<void> {
    try {
      const path: string = await this.copyRawToSandbox(ctx, 'risk_guard.ms');
      const t0: number = Date.now();
      const model = await mindSporeLite.loadModelFromFile(path);
      const loadMs: number = Date.now() - t0;
      const inputs = model.getInputs();
      if (inputs.length < 2) {
        throw new Error('expected 2 inputs, got ' + inputs.length);
      }
      const w0: number = this.fillIntTensor(inputs[0], RG_INPUT_IDS);
      const w1: number = this.fillIntTensor(inputs[1], RG_ATTENTION_MASK);
      this.rgDetail = 'inputs=' + inputs.length +
        ' dtype0=' + inputs[0].dtype + '/' + w0 + 'B' +
        ' dtype1=' + inputs[1].dtype + '/' + w1 + 'B' +
        ' load=' + loadMs + 'ms';

      const t1: number = Date.now();
      const outputs = await model.predict(inputs);
      const inferMs: number = Date.now() - t1;
      const values = new Float32Array(outputs[0].getData());
      let text: string = '';
      let worst: number = 0;
      let argmax: number = 0;
      for (let i = 0; i < values.length; i++) {
        text += (i > 0 ? ', ' : '') + values[i].toFixed(6);
        const d = Math.abs(values[i] - RG_PT_LOGITS[i]);
        if (d > worst) {
          worst = d;
        }
        if (values[i] > values[argmax]) {
          argmax = i;
        }
      }
      this.rgLogits = text;
      this.rgLabel = RG_LABELS[argmax] + ' (pt=' + RG_PT_LABEL + ')';
      this.rgDiff = worst.toExponential(3);
      this.rgOk = worst < 1e-2;
      this.rgStatus = this.rgOk ? 'M2-B PASS' : 'M2-B MISMATCH';
      this.rgDetail = this.rgDetail + ' infer=' + inferMs + 'ms';
      hilog.info(0x0000, TAG,
        'RG_OK sample=%{public}s logits=[%{public}s] label=%{public}s maxdiff=%{public}s load=%{public}dms infer=%{public}dms %{public}s',
        RG_SAMPLE_ID, text, this.rgLabel, this.rgDiff, loadMs, inferMs, this.rgDetail);
    } catch (err) {
      this.rgOk = false;
      this.rgStatus = 'M2-B FAILED';
      this.rgLogits = (err as Error).message;
      hilog.error(0x0000, TAG, 'RG_FAIL %{public}s', (err as Error).message);
    }
  }

  private async runTokenizer(ctx: common.UIAbilityContext): Promise<void> {
    try {
      const vocabBytes: Uint8Array = await ctx.resourceManager.getRawFileContent('vocab.txt');
      const tok = BertTokenizer.load(vocabBytes);
      this.tokenizer = tok;
      let matched: number = 0;
      let firstBad: string = '';
      for (let c = 0; c < TOKENIZER_CASES.length; c++) {
        const item = TOKENIZER_CASES[c];
        const ids: number[] = tok.encode(item.text);
        let same: boolean = ids.length === item.ids.length;
        if (same) {
          for (let i = 0; i < ids.length; i++) {
            if (ids[i] !== item.ids[i]) {
              same = false;
              if (firstBad.length === 0) {
                firstBad = item.id + ' @' + i + ' got=' + ids[i] + ' want=' + item.ids[i];
              }
              break;
            }
          }
        } else if (firstBad.length === 0) {
          firstBad = item.id + ' length ' + ids.length + ' vs ' + item.ids.length;
        }
        if (same) {
          matched++;
        }
      }
      this.tokOk = matched === TOKENIZER_CASES.length;
      this.tokStatus = this.tokOk ? 'M2-C PASS' : 'M2-C MISMATCH';
      this.tokDetail = matched + '/' + TOKENIZER_CASES.length + ' cases match' +
        (firstBad.length > 0 ? ' | first: ' + firstBad : '');
      hilog.info(0x0000, TAG, 'TOK_DONE %{public}s', this.tokDetail);
    } catch (err) {
      this.tokOk = false;
      this.tokStatus = 'M2-C FAILED';
      this.tokDetail = (err as Error).message;
      hilog.error(0x0000, TAG, 'TOK_FAIL %{public}s', (err as Error).message);
    }
  }

  private async runNative(ctx: common.UIAbilityContext): Promise<void> {
    try {
      const raw: Uint8Array = await ctx.resourceManager.getRawFileContent('risk_guard.ms');
      const t0: number = Date.now();
      // Diagnostics: two models x (buffer|file) x (threadNum|default)
      const mlpRaw: Uint8Array = await ctx.resourceManager.getRawFileContent('probe_mlp.ms');
      const diagPath: string = await this.copyRawToSandbox(ctx, 'risk_guard.ms');
      const dMlp: number[] = diagnose(mlpRaw, diagPath);
      const dRg: number[] = diagnose(raw, diagPath);
      hilog.info(0x0000, TAG, 'ND_DIAG mlp=%{public}s rg=%{public}s',
        JSON.stringify(dMlp), JSON.stringify(dRg));
      // Prefer loading from memory; fall back to file-path build if needed.
      let status: number = loadModelFromBuffer(raw);
      let mode: string = 'buffer';
      if (status !== 0) {
        status = loadModel(diagPath);
        mode = 'file';
      }
      const loadMs: number = Date.now() - t0;
      if (status !== 0) {
        throw new Error('load status=' + status + ' (' + mode + ')');
      }
      const t1: number = Date.now();
      const logits: number[] = predict(RG_INPUT_IDS, RG_ATTENTION_MASK);
      const inferMs: number = Date.now() - t1;
      if (logits === null || logits === undefined) {
        throw new Error('predict returned null');
      }
      let text: string = '';
      let worst: number = 0;
      for (let i = 0; i < logits.length; i++) {
        text += (i > 0 ? ', ' : '') + logits[i].toFixed(6);
        const d = Math.abs(logits[i] - RG_PT_LOGITS[i]);
        if (d > worst) {
          worst = d;
        }
      }
      this.ndLogits = text;
      this.ndDiff = worst.toExponential(3);
      this.ndOk = worst < 1e-2;
      this.ndStatus = this.ndOk ? 'M2-D PASS' : 'M2-D MISMATCH';
      this.ndDetail = 'native C API | ' + mode + ' | load=' + loadMs + 'ms infer=' + inferMs + 'ms';
      hilog.info(0x0000, TAG, 'ND_OK logits=[%{public}s] maxdiff=%{public}s %{public}s',
        text, this.ndDiff, this.ndDetail);
    } catch (err) {
      this.ndOk = false;
      this.ndStatus = 'M2-D FAILED';
      this.ndLogits = (err as Error).message;
      hilog.error(0x0000, TAG, 'ND_FAIL %{public}s', (err as Error).message);
    }
  }

  private renderSandbox(): string {
    const b: string = this.sbRecoverable ? '是' : '否';
    return '用户指令：' + this.sbInstruction +
      '；工具：' + this.sbTool +
      '；目标：' + this.sbTarget +
      '；范围：single' +
      '；数量：1' +
      '；授权：' + this.sbAuth +
      '；数据分级：internal' +
      '；可恢复：' + b +
      '；有备份：是' +
      '；目的地信任：' + this.sbTrust;
  }

  private async runSandbox(): Promise<void> {
    this.sbRunning = true;
    this.sbPolicy = '';
    try {
      if (this.tokenizer === null) {
        throw new Error('tokenizer not ready');
      }
      const text: string = this.renderSandbox();
      const ids: number[] = this.tokenizer.encode(text);
      const mask: number[] = [];
      for (let i = 0; i < ids.length; i++) {
        mask.push(ids[i] === 0 ? 0 : 1);
      }
      const logits: number[] = predict(ids, mask);
      if (logits === null || logits === undefined) {
        throw new Error('predict returned null (is the model loaded?)');
      }
      const probs: number[] = softmax(logits);
      this.sbScores = RG_LABELS.map((label: string, i: number) =>
        label + '=' + (probs[i] * 100).toFixed(2) + '%').join('  ');
      let best: number = 0;
      for (let i = 1; i < probs.length; i++) {
        if (probs[i] > probs[best]) {
          best = i;
        }
      }
      const decision: string = RG_LABELS[best];
      this.sbDecision = decision + ' (' + (probs[best] * 100).toFixed(2) + '%)';
      this.sbPolicy = decision === 'BLOCK'
        ? '策略提示：拒绝当前调用并保留审计；不可恢复操作需独立策略引擎二次确认。'
        : (decision === 'ASK'
          ? '策略提示：不自动执行，先向用户补充授权/目的地/恢复条件。'
          : '策略提示：可建议继续，但仍受确定性权限策略约束。');
      hilog.info(0x0000, TAG, 'SB_OK decision=%{public}s scores=%{public}s text=%{public}s',
        decision, this.sbScores, text);
    } catch (err) {
      this.sbDecision = 'FAILED: ' + (err as Error).message;
      hilog.error(0x0000, TAG, 'SB_FAIL %{public}s', (err as Error).message);
    }
    this.sbRunning = false;
  }

  private finishBenchmark(times: number[]): void {
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
    hilog.info(0x0000, TAG, 'BENCH_OK %{public}s', this.benchDetail);
    this.benchRunning = false;
  }

  private benchChunk(ids: number[], mask: number[], times: number[], total: number): void {
    const end: number = Math.min(times.length + 25, total);
    for (let i = times.length; i < end; i++) {
      const t0: number = Date.now();
      predict(ids, mask);
      times.push(Date.now() - t0);
    }
    if (times.length < total) {
      setTimeout(() => {
        this.benchChunk(ids, mask, times, total);
      }, 0);
    } else {
      this.finishBenchmark(times);
    }
  }

  private runBenchmark(): void {
    this.benchRunning = true;
    this.benchStatus = 'running...';
    this.benchDetail = '-';
    try {
      if (this.tokenizer === null) {
        throw new Error('tokenizer not ready');
      }
      const ids: number[] = this.tokenizer.encode(this.renderSandbox());
      const mask: number[] = [];
      for (let i = 0; i < ids.length; i++) {
        mask.push(ids[i] === 0 ? 0 : 1);
      }
      predict(ids, mask); // warmup
      this.benchChunk(ids, mask, [], 1000);
    } catch (err) {
      this.benchStatus = 'M2-F FAILED';
      this.benchDetail = (err as Error).message;
      this.benchRunning = false;
      hilog.error(0x0000, TAG, 'BENCH_FAIL %{public}s', (err as Error).message);
    }
  }

  private async runAll(): Promise<void> {
    const ctx = getContext(this) as common.UIAbilityContext;
    await this.runMlp(ctx);
    await this.runRiskGuard(ctx);
    await this.runTokenizer(ctx);
    await this.runNative(ctx);
  }

  build() {
    Scroll() {
      Column({ space: 8 }) {
        Text('Harmony Risk Guard · M2 Probe')
          .fontSize(22)
          .fontWeight(FontWeight.Bold)
          .margin({ bottom: 6 })

        Text('— M2-A: MLP 4-16-3 —').fontSize(16).fontWeight(FontWeight.Medium)
        Text(this.mlpStatus)
          .fontSize(16)
          .fontColor(this.mlpOk ? '#1B7F3A' : '#B3261E')
        Text('logits: ' + this.mlpLogits).fontSize(13)
        Text('max |Δ| vs PyTorch: ' + this.mlpDiff).fontSize(13)

        Text('— M2-B: MiniRBT risk classifier —').fontSize(16)
          .fontWeight(FontWeight.Medium).margin({ top: 12 })
        Text(this.rgStatus)
          .fontSize(16)
          .fontColor(this.rgOk ? '#1B7F3A' : '#B3261E')
        Text('sample: ' + RG_SAMPLE_ID).fontSize(13)
        Text('logits: ' + this.rgLogits).fontSize(13)
        Text('expected: ' + RG_PT_LOGITS.join(', ')).fontSize(13).fontColor('#666666')
        Text('label: ' + this.rgLabel).fontSize(13)
        Text('max |Δ| vs PyTorch: ' + this.rgDiff).fontSize(13)
        Text(this.rgDetail).fontSize(12).fontColor('#666666')

        Text('— M2-C: Tokenizer (WordPiece) —').fontSize(16)
          .fontWeight(FontWeight.Medium).margin({ top: 12 })
        Text(this.tokStatus)
          .fontSize(16)
          .fontColor(this.tokOk ? '#1B7F3A' : '#B3261E')
        Text(this.tokDetail).fontSize(13)

        Text('— M2-D: Native NAPI (C API) —').fontSize(16)
          .fontWeight(FontWeight.Medium).margin({ top: 12 })
        Text(this.ndStatus)
          .fontSize(16)
          .fontColor(this.ndOk ? '#1B7F3A' : '#B3261E')
        Text('logits: ' + this.ndLogits).fontSize(13)
        Text('max |Δ| vs PyTorch: ' + this.ndDiff).fontSize(13)
        Text(this.ndDetail).fontSize(12).fontColor('#666666')

        Divider().margin({ top: 18, bottom: 6 })
        Text('— M2-E: 交互式风险判定 —').fontSize(16)
          .fontWeight(FontWeight.Medium)
        TextInput({ text: this.sbInstruction, placeholder: '用户指令' })
          .fontSize(13).margin({ top: 6 })
          .onChange((v: string) => { this.sbInstruction = v; })
        TextInput({ text: this.sbTool, placeholder: '工具 (如 file.delete)' })
          .fontSize(13).margin({ top: 4 })
          .onChange((v: string) => { this.sbTool = v; })
        TextInput({ text: this.sbTarget, placeholder: '目标类型 (如 user_document)' })
          .fontSize(13).margin({ top: 4 })
          .onChange((v: string) => { this.sbTarget = v; })
        Row({ space: 8 }) {
          Text('授权:').fontSize(13)
          Select([{ value: 'approved' }, { value: 'ambiguous' }, { value: 'denied' }])
            .selected(this.sbAuth === 'approved' ? 0 : (this.sbAuth === 'ambiguous' ? 1 : 2))
            .value(this.sbAuth)
            .onSelect((index: number) => {
              this.sbAuth = index === 0 ? 'approved' : (index === 1 ? 'ambiguous' : 'denied');
            })
          Checkbox().select(this.sbRecoverable)
            .onChange((v: boolean) => { this.sbRecoverable = v; })
          Text('可恢复').fontSize(13)
          Text('目的地:').fontSize(13)
          Select([{ value: 'trusted' }, { value: 'untrusted' }])
            .selected(this.sbTrust === 'trusted' ? 0 : 1)
            .value(this.sbTrust)
            .onSelect((index: number) => {
              this.sbTrust = index === 0 ? 'trusted' : 'untrusted';
            })
        }.margin({ top: 8 }).width('100%')
        Button(this.sbRunning ? '运行中...' : '运行判定')
          .enabled(!this.sbRunning)
          .onClick(() => { this.runSandbox(); })
          .margin({ top: 8 })
        Text('分数: ' + this.sbScores).fontSize(13)
        Text('决策: ' + this.sbDecision).fontSize(15).fontWeight(FontWeight.Medium)
        Text(this.sbPolicy).fontSize(12).fontColor('#666666')

        Text('— M2-F: 基准 1000 次 —').fontSize(16)
          .fontWeight(FontWeight.Medium).margin({ top: 14 })
        Button(this.benchRunning ? '基准运行中...' : '运行 1000 次基准')
          .enabled(!this.benchRunning)
          .onClick(() => { this.runBenchmark(); })
          .margin({ top: 6 })
        Text(this.benchStatus).fontSize(14)
        Text(this.benchDetail).fontSize(12).fontColor('#666666')
      }
      .padding(24)
      .width('100%')
    }
    .width('100%')
    .height('100%')
  }
}
