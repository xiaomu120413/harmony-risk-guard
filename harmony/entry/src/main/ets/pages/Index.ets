import { mindSporeLite } from '@kit.MindSporeLiteKit';
import { hilog } from '@kit.PerformanceAnalysisKit';
import { fileIo } from '@kit.CoreFileKit';
import { common } from '@kit.AbilityKit';

const TAG: string = 'RiskGuardProbe';
// PyTorch CPU reference for the same weights (scripts/make_probe_mlp.py, seed=42)
const GOLDEN: number[] = [0.161677, -0.013502, -0.003518];
const SAMPLE: number[] = [0.1, 0.2, 0.3, 0.4];

@Entry
@Component
struct Index {
  @State private status: string = 'running...';
  @State private ok: boolean = false;
  @State private modelBytes: number = 0;
  @State private logits: string = '-';
  @State private maxDiff: string = '-';
  @State private detail: string = '';

  aboutToAppear(): void {
    this.runProbe();
  }

  private async runProbe(): Promise<void> {
    try {
      const ctx = getContext(this) as common.UIAbilityContext;
      // 1) copy rawfile probe_mlp.ms into the app sandbox
      const raw: Uint8Array = await ctx.resourceManager.getRawFileContent('probe_mlp.ms');
      this.modelBytes = raw.byteLength;
      const path: string = ctx.filesDir + '/probe_mlp.ms';
      const file = fileIo.openSync(path,
        fileIo.OpenMode.CREATE | fileIo.OpenMode.READ_WRITE | fileIo.OpenMode.TRUNC);
      fileIo.writeSync(file.fd, raw.buffer);
      fileIo.closeSync(file);
      hilog.info(0x0000, TAG, 'model copied: %{public}d bytes -> %{public}s', raw.byteLength, path);

      // 2) load model with MindSpore Lite
      const model = await mindSporeLite.loadModelFromFile(path);
      const inputs = model.getInputs();
      this.detail = 'inputs=' + inputs.length + ' name=' + inputs[0].name +
        ' shape=' + JSON.stringify(inputs[0].shape);
      hilog.info(0x0000, TAG, 'MODEL_LOADED %{public}s', this.detail);

      // 3) fixed sample input -> predict
      const x = new Float32Array(SAMPLE);
      inputs[0].setData(x.buffer);
      const outputs = await model.predict(inputs);
      const out = outputs[0];
      const values = new Float32Array(out.getData());

      let text: string = '';
      let worst: number = 0;
      for (let i = 0; i < values.length; i++) {
        text += (i > 0 ? ', ' : '') + values[i].toFixed(6);
        const d = Math.abs(values[i] - GOLDEN[i]);
        if (d > worst) {
          worst = d;
        }
      }
      this.logits = text;
      this.maxDiff = worst.toExponential(3);
      this.ok = worst < 1e-3;
      this.status = this.ok ? 'M2-A PROBE PASS' : 'M2-A PROBE MISMATCH';
      hilog.info(0x0000, TAG, 'PROBE_OK logits=[%{public}s] maxdiff=%{public}s',
        text, this.maxDiff);
    } catch (err) {
      const e = err as Error;
      this.ok = false;
      this.status = 'M2-A PROBE FAILED';
      this.logits = '-';
      this.detail = e.message;
      hilog.error(0x0000, TAG, 'PROBE_FAIL %{public}s', e.message);
    }
  }

  build() {
    Column({ space: 10 }) {
      Text('Harmony Risk Guard · M2-A Probe')
        .fontSize(22)
        .fontWeight(FontWeight.Bold)
        .margin({ bottom: 8 })
      Text(this.status)
        .fontSize(18)
        .fontWeight(FontWeight.Medium)
        .fontColor(this.ok ? '#1B7F3A' : '#B3261E')
      Text('model: probe_mlp.ms (' + this.modelBytes + ' bytes)').fontSize(14)
      Text('input: [0.1, 0.2, 0.3, 0.4]').fontSize(14)
      Text('logits: ' + this.logits).fontSize(14)
      Text('expected(logits): 0.161677, -0.013502, -0.003518').fontSize(14).fontColor('#666666')
      Text('max |Δ| vs PyTorch: ' + this.maxDiff).fontSize(14)
      Text(this.detail).fontSize(12).fontColor('#666666')
    }
    .padding(24)
    .width('100%')
    .height('100%')
    .justifyContent(FlexAlign.Center)
  }
}
