import { common } from '@kit.AbilityKit';
import { preferences } from '@kit.ArkData';

export interface HistoryEntry {
  time: string;
  instruction: string;
  tool: string;
  target: string;
  decision: string;
  scores: string;
}

const STORE_NAME: string = 'riskguard_store';
const KEY_HISTORY: string = 'history_v1';
const MAX_ENTRIES: number = 20;

export function nowText(): string {
  const d = new Date();
  const pad = (n: number): string => n < 10 ? '0' + n : '' + n;
  return d.getFullYear() + '-' + pad(d.getMonth() + 1) + '-' + pad(d.getDate()) +
    ' ' + pad(d.getHours()) + ':' + pad(d.getMinutes());
}

export class HistoryStore {
  static async list(ctx: common.UIAbilityContext): Promise<HistoryEntry[]> {
    try {
      const store = await preferences.getPreferences(ctx, STORE_NAME);
      const raw = await store.get(KEY_HISTORY, '[]');
      return JSON.parse(raw as string) as HistoryEntry[];
    } catch (err) {
      return [];
    }
  }

  static async add(ctx: common.UIAbilityContext, entry: HistoryEntry): Promise<void> {
    const list: HistoryEntry[] = await HistoryStore.list(ctx);
    list.unshift(entry);
    const trimmed: HistoryEntry[] = list.slice(0, MAX_ENTRIES);
    await HistoryStore.save(ctx, trimmed);
  }

  static async clear(ctx: common.UIAbilityContext): Promise<void> {
    await HistoryStore.save(ctx, []);
  }

  private static async save(ctx: common.UIAbilityContext, list: HistoryEntry[]): Promise<void> {
    const store = await preferences.getPreferences(ctx, STORE_NAME);
    await store.put(KEY_HISTORY, JSON.stringify(list));
    await store.flush();
  }
}
