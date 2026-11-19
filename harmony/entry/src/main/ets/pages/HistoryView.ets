import { common } from '@kit.AbilityKit';
import { HistoryStore, HistoryEntry } from '../services/HistoryStore';
import { decisionText } from '../services/RiskGuardService';

function decisionColor(decision: string): string {
  if (decision === 'CONTINUE') {
    return '#1B7F3A';
  }
  if (decision === 'ASK') {
    return '#B26A00';
  }
  return '#D33A31';
}

@Component
export struct HistoryView {
  @Prop @Watch('onRefresh') refreshTick: number = 0;
  @State private entries: HistoryEntry[] = [];
  @State private loaded: boolean = false;

  aboutToAppear(): void {
    this.load();
  }

  onRefresh(): void {
    this.load();
  }

  private async load(): Promise<void> {
    const ctx = getContext(this) as common.UIAbilityContext;
    this.entries = await HistoryStore.list(ctx);
    this.loaded = true;
  }

  private async clearAll(): Promise<void> {
    const ctx = getContext(this) as common.UIAbilityContext;
    await HistoryStore.clear(ctx);
    this.entries = [];
  }

  build() {
    Column() {
      Row() {
        Column({ space: 4 }) {
          Text('评估历史').fontSize(24).fontWeight(FontWeight.Bold).fontColor('#182431')
          Text('最近 20 条 · 仅保存在本机').fontSize(13).fontColor('#6B7785')
        }.alignItems(HorizontalAlign.Start)
        Blank()
        Button('清空记录')
          .fontSize(13)
          .height(34)
          .backgroundColor('#FFFFFF')
          .fontColor('#D33A31')
          .border({ width: 1, color: '#F0D2D0' })
          .enabled(this.entries.length > 0)
          .onClick(() => { this.clearAll(); })
      }.width('100%').margin({ bottom: 16 })

      if (this.loaded && this.entries.length === 0) {
        Column({ space: 8 }) {
          Text('暂无评估记录').fontSize(16).fontColor('#6B7785')
          Text('在"风险评估"页完成一次评估后，结果会显示在这里').fontSize(12).fontColor('#999999')
        }
        .layoutWeight(1)
        .width('100%')
        .justifyContent(FlexAlign.Center)
      } else {
        List({ space: 10 }) {
          ForEach(this.entries, (item: HistoryEntry, index: number) => {
            ListItem() {
              Row({ space: 14 }) {
                Text(decisionText(item.decision))
                  .fontSize(13)
                  .fontWeight(FontWeight.Medium)
                  .fontColor('#FFFFFF')
                  .backgroundColor(decisionColor(item.decision))
                  .borderRadius(8)
                  .padding({ left: 12, right: 12, top: 5, bottom: 5 })
                Column({ space: 4 }) {
                  Text(item.instruction)
                    .fontSize(14)
                    .fontColor('#182431')
                    .maxLines(1)
                    .textOverflow({ overflow: TextOverflow.Ellipsis })
                  Text(item.time + ' · ' + item.tool + ' → ' + item.target)
                    .fontSize(11)
                    .fontColor('#8A94A6')
                    .maxLines(1)
                    .textOverflow({ overflow: TextOverflow.Ellipsis })
                }.alignItems(HorizontalAlign.Start).layoutWeight(1)
                Text(item.scores).fontSize(11).fontColor('#8A94A6')
              }
              .padding(14)
              .backgroundColor('#FFFFFF')
              .borderRadius(12)
              .width('100%')
            }
          }, (item: HistoryEntry, index: number) => String(index) + item.time)
        }
        .layoutWeight(1)
        .width('100%')
      }
    }
    .alignItems(HorizontalAlign.Start)
    .padding({ left: 28, right: 28, top: 26, bottom: 30 })
    .width('100%')
    .height('100%')
    .backgroundColor('#F1F3F5')
  }
}
