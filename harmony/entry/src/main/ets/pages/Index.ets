import { AssessView } from './AssessView';
import { HistoryView } from './HistoryView';
import { AboutView } from './AboutView';

@Entry
@Component
struct Index {
  @State private selectedIndex: number = 0;
  @State private refreshTick: number = 0;

  private select(index: number): void {
    this.selectedIndex = index;
    if (index === 1) {
      this.refreshTick = this.refreshTick + 1;
    }
  }

  @Builder
  navItem(label: string, index: number) {
    Row({ space: 10 }) {
      Row()
        .width(6)
        .height(6)
        .borderRadius(3)
        .backgroundColor(this.selectedIndex === index ? '#0A59F7' : '#C9D0DA')
      Text(label)
        .fontSize(14)
        .fontColor(this.selectedIndex === index ? '#0A59F7' : '#4A5568')
        .fontWeight(this.selectedIndex === index ? FontWeight.Medium : FontWeight.Normal)
      Blank()
      if (this.selectedIndex === index) {
        Row().width(3).height(16).backgroundColor('#0A59F7').borderRadius(2)
      }
    }
    .width('100%')
    .padding({ left: 14, right: 14, top: 11, bottom: 11 })
    .backgroundColor(this.selectedIndex === index ? '#E8F0FE' : Color.Transparent)
    .borderRadius(10)
    .onClick(() => { this.select(index); })
  }

  build() {
    Row() {
      // Sidebar
      Column() {
        Row({ space: 10 }) {
          Image($r('app.media.startIcon')).width(38).height(38).borderRadius(10)
          Column({ space: 1 }) {
            Text('风险守卫').fontSize(16).fontWeight(FontWeight.Bold).fontColor('#182431')
            Text('本地离线评估').fontSize(11).fontColor('#8A94A6')
          }.alignItems(HorizontalAlign.Start)
        }
        .width('100%')
        .padding({ left: 16, right: 16, top: 22, bottom: 18 })

        Column({ space: 6 }) {
          this.navItem('风险评估', 0)
          this.navItem('历史记录', 1)
          this.navItem('关于', 2)
        }
        .padding({ left: 10, right: 10 })
        .width('100%')

        Blank()
        Text('v1.0.0 · 模型不出设备')
          .fontSize(10)
          .fontColor('#A0A8B8')
          .padding(16)
      }
      .width(208)
      .height('100%')
      .backgroundColor('#FFFFFF')
      .border({ width: { right: 1 }, color: '#E8EAEE' })

      // Content (all pages stay mounted; Visibility keeps state)
      Stack() {
        Column() {
          AssessView()
        }
        .width('100%')
        .height('100%')
        .visibility(this.selectedIndex === 0 ? Visibility.Visible : Visibility.None)

        Column() {
          HistoryView({ refreshTick: this.refreshTick })
        }
        .width('100%')
        .height('100%')
        .visibility(this.selectedIndex === 1 ? Visibility.Visible : Visibility.None)

        Column() {
          AboutView()
        }
        .width('100%')
        .height('100%')
        .visibility(this.selectedIndex === 2 ? Visibility.Visible : Visibility.None)
      }
      .layoutWeight(1)
      .height('100%')
      .backgroundColor('#F1F3F5')
    }
    .width('100%')
    .height('100%')
    .backgroundColor('#F1F3F5')
  }
}
