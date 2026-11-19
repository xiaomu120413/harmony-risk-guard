# M2-C：端侧 Tokenizer 一致性 — PASS

日期：2026-11-19 · 设备：HUAWEI MateBook Pro（HAD-W32）/ OpenHarmony 7.1 / API 26
实现：`harmony/entry/src/main/ets/model/BertTokenizer.ets`（纯 ArkTS WordPiece）

## 1. 结论

**30/30 条 golden cases 的 token IDs 与 Windows 侧 HuggingFace BertTokenizer 完全一致。**

设备日志：

```
TOK_DONE 30/30 cases match
```

UI（[截图](evidence/m2c_all_probes_pass.jpeg)）：M2-A / M2-B / M2-C 三项同屏 PASS。

## 2. 实现要点

严格复刻 HuggingFace `BertTokenizer(do_lower_case=True)`：

1. `clean_text`：控制字符/空白归一
2. `tokenize_chinese_chars`：CJK 表意字符两侧加空格（BMP 范围 0x4E00-0x9FFF /
   0x3400-0x4DBF / 0xF900-0xFAFF；数据集文本均 BMP）
3. 按空白切词 → 小写 → 标点拆分（ASCII 标点 + CJK/全角标点区间近似
   `unicodedata.category('P*')`）
4. WordPiece 贪心最长匹配（`##` 续接、单 word 最多 100 字符、未命中 → `[UNK]`）
5. `[CLS] + tokens + [SEP]`，截断到 128，`[PAD]` 补齐

验证方式：
- 先在 Windows 侧用 Python 复刻同一算法自检 → 30/30 匹配（排除逻辑错误）
- 再在设备上跑 ArkTS 实现 → 30/30 匹配

## 3. 踩坑（已修复）

| 现象 | 根因 | 修复 |
|---|---|---|
| 首次实机 `TOK_DONE 0/30`，`@1 got=100([UNK]) want=4500` | `.ms` 读取正常但 vocab.txt 是 **UTF-8**，初版按单字节 Latin-1 解码 → 中文词条全部乱码 | 用 `util.TextDecoder.create('utf-8')` + `decodeToString` 解码后再建词表 |

## 4. 产物

| 路径 | 内容 |
|---|---|
| `harmony/entry/src/main/ets/model/BertTokenizer.ets` | ArkTS WordPiece 实现 |
| `harmony/entry/src/main/ets/model/TokenizerGolden.ets` | 30 条 golden cases（text + ids[128] + mask[128]） |
| `harmony/entry/src/main/resources/rawfile/vocab.txt` | MiniRBT 词表（21,128 条） |
| `scripts/gen_tokenizer_golden.py` | golden 数据生成器 |
| `docs/evidence/m2c_all_probes_pass.jpeg` | 三项 PASS 同屏截图 |
