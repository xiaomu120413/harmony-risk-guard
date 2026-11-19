import { common } from '@kit.AbilityKit';
import { loadModelFromBuffer, predict } from 'libriskguard_native.so';
import { BertTokenizer } from '../model/BertTokenizer';

export const DECISION_LABELS: string[] = ['CONTINUE', 'ASK', 'BLOCK'];

export function decisionText(decision: string): string {
  if (decision === 'CONTINUE') {
    return '可继续';
  }
  if (decision === 'ASK') {
    return '需确认';
  }
  return '应阻止';
}

export interface AssessInput {
  instruction: string;
  tool: string;
  target: string;
  scope: string;
  affectedCount: number;
  authorization: string;
  classification: string;
  recoverable: boolean;
  backupAvailable: boolean;
  destinationTrust: string;
}

export interface AssessResult {
  decision: string;
  decisionIndex: number;
  continueScore: number;
  askScore: number;
  blockScore: number;
  latencyMs: number;
  serialized: string;
}

function zhBool(v: boolean): string {
  return v ? '是' : '否';
}

export function renderInput(input: AssessInput): string {
  return '用户指令：' + input.instruction +
    '；工具：' + input.tool +
    '；目标：' + input.target +
    '；范围：' + input.scope +
    '；数量：' + input.affectedCount +
    '；授权：' + input.authorization +
    '；数据分级：' + input.classification +
    '；可恢复：' + zhBool(input.recoverable) +
    '；有备份：' + zhBool(input.backupAvailable) +
    '；目的地信任：' + input.destinationTrust;
}

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

/**
 * Singleton owning the on-device model + tokenizer.
 * Model is loaded once per process; UI calls ensureReady() during page appearance
 * and assess() on demand.
 */
export class RiskGuardService {
  private static instance: RiskGuardService | null = null;
  private tokenizer: BertTokenizer | null = null;
  private modelReady: boolean = false;
  private initPromise: Promise<void> | null = null;
  private loadMs: number = 0;
  private lastError: string = '';

  static get(): RiskGuardService {
    if (RiskGuardService.instance === null) {
      RiskGuardService.instance = new RiskGuardService();
    }
    return RiskGuardService.instance;
  }

  isReady(): boolean {
    return this.modelReady;
  }

  loadTimeMs(): number {
    return this.loadMs;
  }

  error(): string {
    return this.lastError;
  }

  ensureReady(ctx: common.UIAbilityContext): Promise<void> {
    if (this.modelReady) {
      return Promise.resolve();
    }
    if (this.initPromise !== null) {
      return this.initPromise;
    }
    this.initPromise = this.doInit(ctx);
    this.initPromise.then(() => {
      this.initPromise = null;
    }).catch(() => {
      this.initPromise = null;
    });
    return this.initPromise;
  }

  private async doInit(ctx: common.UIAbilityContext): Promise<void> {
    try {
      const vocabBytes: Uint8Array = await ctx.resourceManager.getRawFileContent('vocab.txt');
      this.tokenizer = BertTokenizer.load(vocabBytes);
      const modelBytes: Uint8Array = await ctx.resourceManager.getRawFileContent('risk_guard.ms');
      const t0: number = Date.now();
      const status: number = loadModelFromBuffer(modelBytes);
      if (status !== 0) {
        throw new Error('模型加载失败（status=' + status + '）');
      }
      this.loadMs = Date.now() - t0;
      this.modelReady = true;
      this.lastError = '';
    } catch (err) {
      this.modelReady = false;
      this.lastError = (err as Error).message;
      throw err as Error;
    }
  }

  async assess(ctx: common.UIAbilityContext, input: AssessInput): Promise<AssessResult> {
    await this.ensureReady(ctx);
    if (this.tokenizer === null) {
      throw new Error('分词器未初始化');
    }
    const serialized: string = renderInput(input);
    const ids: number[] = this.tokenizer.encode(serialized);
    const mask: number[] = [];
    for (let i = 0; i < ids.length; i++) {
      mask.push(ids[i] === 0 ? 0 : 1);
    }
    const t0: number = Date.now();
    const logits: number[] = predict(ids, mask);
    const latencyMs: number = Date.now() - t0;
    if (logits === null || logits === undefined) {
      throw new Error('推理失败，请重试');
    }
    const probs: number[] = softmax(logits);
    let best: number = 0;
    for (let i = 1; i < probs.length; i++) {
      if (probs[i] > probs[best]) {
        best = i;
      }
    }
    return {
      decision: DECISION_LABELS[best],
      decisionIndex: best,
      continueScore: probs[0],
      askScore: probs[1],
      blockScore: probs[2],
      latencyMs: latencyMs,
      serialized: serialized
    };
  }
}
