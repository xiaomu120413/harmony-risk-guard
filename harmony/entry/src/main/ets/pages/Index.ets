import { mindSporeLite } from '@kit.MindSporeLiteKit';
import { hilog } from '@kit.PerformanceAnalysisKit';
import { fileIo } from '@kit.CoreFileKit';
import { common } from '@kit.AbilityKit';
import {
  RG_SAMPLE_ID, RG_INPUT_IDS, RG_ATTENTION_MASK, RG_PT_LOGITS, RG_PT_LABEL, RG_LABELS
} from '../model/GoldenData';
import { TOKENIZER_CASES } from '../model/TokenizerGolden';
import { BertTokenizer } from '../model/BertTokenizer';

const TAG: string = 'RiskGuardProbe';
// M2-A probe: PyTorch CPU reference for the 4-16-3 MLP (seed=42)
const MLP_GOLDEN: number[] = [0.161677, -0.013502, -0.003518];
const MLP_SAMPLE: number[] = [0.1, 0.2, 0.3, 0.4];

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

  private async runAll(): Promise<void> {
    const ctx = getContext(this) as common.UIAbilityContext;
    await this.runMlp(ctx);
    await this.runRiskGuard(ctx);
    await this.runTokenizer(ctx);
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
      }
      .padding(24)
      .width('100%')
    }
    .width('100%')
    .height('100%')
  }
}
