import { router } from '@kit.ArkUI';

@Component
export struct AboutView {
  @Builder
  infoRow(label: string, value: string) {
    Row({ space: 12 }) {
      Text(label).fontSize(13).fontColor('#6B7785').width(96)
      Text(value).fontSize(13).fontColor('#182431').layoutWeight(1)
    }.width('100%').padding({ top: 6, bottom: 6 })
  }

  build() {
    Scroll() {
      Column({ space: 14 }) {
        Row() {
          Column({ space: 4 }) {
            Text('关于风险守卫').fontSize(24).fontWeight(FontWeight.Bold).fontColor('#182431')
            Text('Agent 操作风险评估工具 · 端侧离线版').fontSize(13).fontColor('#6B7785')
          }.alignItems(HorizontalAlign.Start)
          Blank()
          Button('自检与诊断')
            .fontSize(13)
            .height(34)
            .backgroundColor('#FFFFFF')
            .fontColor('#0A59F7')
            .border({ width: 1, color: '#BFD3F2' })
            .onClick(() => {
              router.pushUrl({ url: 'pages/Diagnostics' });
            })
        }.width('100%')

        Column({ space: 2 }) {
          Text('应用信息').fontSize(15).fontWeight(FontWeight.Medium).fontColor('#182431')
            .margin({ bottom: 6 })
          this.infoRow('版本', '1.0.0')
          this.infoRow('运行平台', 'HarmonyOS PC（2in1）')
          this.infoRow('网络', '完全离线，不联网')
        }
        .alignItems(HorizontalAlign.Start)
        .padding(20).backgroundColor('#FFFFFF').borderRadius(16).width('100%')

        Column({ space: 2 }) {
          Text('模型信息').fontSize(15).fontWeight(FontWeight.Medium).fontColor('#182431')
            .margin({ bottom: 6 })
          this.infoRow('模型', 'MiniRBT-h256 编码器 + 三分类头')
          this.infoRow('大小', '41.2 MB（MindSpore Lite .ms）')
          this.infoRow('输入', '指令 / 工具 / 目标 / 上下文（token ids [1,128]）')
          this.infoRow('输出', '可继续 CONTINUE · 需确认 ASK · 应阻止 BLOCK')
          this.infoRow('推理', 'CPU 4 线程（MindSpore Lite，未启用 NPU/GPU），单次约 9–15 ms')
        }
        .alignItems(HorizontalAlign.Start)
        .padding(20).backgroundColor('#FFFFFF').borderRadius(16).width('100%')

        Column({ space: 8 }) {
          Text('隐私说明').fontSize(15).fontWeight(FontWeight.Medium).fontColor('#182431')
          Text('所有评估均在本机完成：输入内容、模型推理与历史记录都不会离开设备，应用不申请网络权限。')
            .fontSize(13).fontColor('#4A5568').width('100%')
        }
        .alignItems(HorizontalAlign.Start)
        .padding(20).backgroundColor('#FFFFFF').borderRadius(16).width('100%')

        Column({ space: 8 }) {
          Text('免责声明').fontSize(15).fontWeight(FontWeight.Medium).fontColor('#B26A00')
          Text('本工具输出的只是风险建议，不构成授权，也不能替代系统权限校验。' +
            '模型基于合成数据训练，未在真实业务环境中验证；' +
            '涉及不可恢复删除、凭据外发等高风险操作时，必须由确定性策略引擎独立判断并在不确定时拒绝执行（fail closed）。')
            .fontSize(13).fontColor('#4A5568').width('100%')
        }
        .alignItems(HorizontalAlign.Start)
        .padding(20).backgroundColor('#FFFFFF').borderRadius(16).width('100%')

        Text('本页面信息仅供产品说明；自检诊断页用于验证模型与运行时完整性。')
          .fontSize(11).fontColor('#999999').width('100%').margin({ top: 2 })
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
