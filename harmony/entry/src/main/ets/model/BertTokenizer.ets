// BertTokenizer (WordPiece) for MiniRBT — faithful port of HuggingFace BertTokenizer
// (do_lower_case=true). Verified against 30 golden cases produced on Windows.
// Limits: BMP CJK ranges only (all dataset text is BMP); accent stripping omitted
// (dataset text contains no accented Latin characters).

import { util } from '@kit.ArkTS';

const UNK_TOKEN: string = '[UNK]';
const CLS_TOKEN: string = '[CLS]';
const SEP_TOKEN: string = '[SEP]';
const PAD_TOKEN: string = '[PAD]';
const MAX_INPUT_CHARS_PER_WORD: number = 100;

export class BertTokenizer {
  private vocab: Map<string, number> = new Map<string, number>();
  private clsId: number = 101;
  private sepId: number = 102;
  private unkId: number = 100;
  private padId: number = 0;
  private maxLen: number = 128;
  private lower: boolean = true;

  static load(vocabBytes: Uint8Array): BertTokenizer {
    const t = new BertTokenizer();
    // vocab.txt is UTF-8 (Chinese entries are multi-byte): decode properly.
    const decoder = util.TextDecoder.create('utf-8');
    const text: string = decoder.decodeToString(vocabBytes);
    const lines: string[] = text.split('\n');
    let idx: number = 0;
    for (let i = 0; i < lines.length; i++) {
      let line: string = lines[i];
      if (line.endsWith('\r')) {
        line = line.substring(0, line.length - 1);
      }
      // A trailing newline at EOF produces one final empty element; do not assign an id to it.
      if (i === lines.length - 1 && line.length === 0) {
        break;
      }
      t.vocab.set(line, idx);
      idx++;
    }
    // resolve special ids from vocab (fallback to BERT defaults)
    const cls: number | undefined = t.vocab.get(CLS_TOKEN);
    const sep: number | undefined = t.vocab.get(SEP_TOKEN);
    const unk: number | undefined = t.vocab.get(UNK_TOKEN);
    const pad: number | undefined = t.vocab.get(PAD_TOKEN);
    if (cls !== undefined) {
      t.clsId = cls;
    }
    if (sep !== undefined) {
      t.sepId = sep;
    }
    if (unk !== undefined) {
      t.unkId = unk;
    }
    if (pad !== undefined) {
      t.padId = pad;
    }
    return t;
  }

  private static isWhitespace(code: number): boolean {
    return code === 32 || code === 9 || code === 10 || code === 13;
  }

  private static isControl(code: number): boolean {
    if (code === 9 || code === 10 || code === 13) {
      return false;
    }
    return code < 32 || code === 127;
  }

  private static isCjk(code: number): boolean {
    return (code >= 0x4E00 && code <= 0x9FFF) ||
      (code >= 0x3400 && code <= 0x4DBF) ||
      (code >= 0xF900 && code <= 0xFAFF);
  }

  private static isPunctuation(code: number): boolean {
    // ASCII punctuation ranges (same as HF _is_punctuation)
    if ((code >= 33 && code <= 47) || (code >= 58 && code <= 64) ||
      (code >= 91 && code <= 96) || (code >= 123 && code <= 126)) {
      return true;
    }
    // Unicode punctuation approximations covering CJK/fullwidth forms in this dataset:
    if (code >= 0x00A1 && code <= 0x00BF) {
      return true;
    }
    if (code >= 0x2000 && code <= 0x206F) {
      return true;
    }
    if (code >= 0x3001 && code <= 0x3003) {
      return true;
    }
    if (code >= 0x3008 && code <= 0x3011) {
      return true;
    }
    if (code >= 0x3014 && code <= 0x301F) {
      return true;
    }
    if (code === 0x30FB) {
      return true;
    }
    if (code >= 0x2E00 && code <= 0x2E7F) {
      return true;
    }
    // fullwidth ASCII punctuation: ！＂＃…（），：；？ etc.
    if ((code >= 0xFF01 && code <= 0xFF0F) || (code >= 0xFF1A && code <= 0xFF20) ||
      (code >= 0xFF3B && code <= 0xFF40) || (code >= 0xFF5B && code <= 0xFF65)) {
      return true;
    }
    return false;
  }

  private cleanText(text: string): string {
    let out: string = '';
    for (let i = 0; i < text.length; i++) {
      const c: number = text.charCodeAt(i);
      if (BertTokenizer.isWhitespace(c)) {
        out += ' ';
      } else if (!BertTokenizer.isControl(c)) {
        out += text.charAt(i);
      }
    }
    // tokenize_chinese_chars: pad CJK ideographs with spaces
    let spaced: string = '';
    for (let i = 0; i < out.length; i++) {
      const c: number = out.charCodeAt(i);
      if (BertTokenizer.isCjk(c)) {
        spaced += ' ' + out.charAt(i) + ' ';
      } else {
        spaced += out.charAt(i);
      }
    }
    return spaced;
  }

  private static whitespaceTokenize(text: string): string[] {
    const parts: string[] = [];
    let cur: string = '';
    for (let i = 0; i < text.length; i++) {
      const c: number = text.charCodeAt(i);
      if (BertTokenizer.isWhitespace(c)) {
        if (cur.length > 0) {
          parts.push(cur);
          cur = '';
        }
      } else {
        cur += text.charAt(i);
      }
    }
    if (cur.length > 0) {
      parts.push(cur);
    }
    return parts;
  }

  private static splitOnPunc(token: string): string[] {
    const out: string[] = [];
    let cur: string = '';
    for (let i = 0; i < token.length; i++) {
      const c: number = token.charCodeAt(i);
      if (BertTokenizer.isPunctuation(c)) {
        if (cur.length > 0) {
          out.push(cur);
          cur = '';
        }
        out.push(token.charAt(i));
      } else {
        cur += token.charAt(i);
      }
    }
    if (cur.length > 0) {
      out.push(cur);
    }
    return out;
  }

  private basicTokenize(text: string): string[] {
    const result: string[] = [];
    const cleaned: string = this.cleanText(text);
    const words: string[] = BertTokenizer.whitespaceTokenize(cleaned);
    for (let i = 0; i < words.length; i++) {
      let w: string = words[i];
      if (this.lower) {
        w = w.toLowerCase();
      }
      const subTokens: string[] = BertTokenizer.splitOnPunc(w);
      for (let j = 0; j < subTokens.length; j++) {
        result.push(subTokens[j]);
      }
    }
    return result;
  }

  private wordPiece(word: string): string[] {
    if (word.length > MAX_INPUT_CHARS_PER_WORD) {
      return [UNK_TOKEN];
    }
    const subTokens: string[] = [];
    let start: number = 0;
    while (start < word.length) {
      let end: number = word.length;
      let cur: string = '';
      let found: boolean = false;
      while (start < end) {
        let piece: string = word.substring(start, end);
        if (start > 0) {
          piece = '##' + piece;
        }
        if (this.vocab.has(piece)) {
          cur = piece;
          found = true;
          break;
        }
        end--;
      }
      if (!found) {
        return [UNK_TOKEN];
      }
      subTokens.push(cur);
      start = end;
    }
    return subTokens;
  }

  private idOf(token: string): number {
    const id: number | undefined = this.vocab.get(token);
    return id === undefined ? this.unkId : id;
  }

  encode(text: string): number[] {
    const words: string[] = this.basicTokenize(text);
    let ids: number[] = [this.clsId];
    for (let i = 0; i < words.length; i++) {
      const pieces: string[] = this.wordPiece(words[i]);
      for (let j = 0; j < pieces.length; j++) {
        ids.push(this.idOf(pieces[j]));
      }
    }
    ids.push(this.sepId);
    if (ids.length > this.maxLen) {
      ids = ids.slice(0, this.maxLen);
      ids[this.maxLen - 1] = this.sepId;
    }
    while (ids.length < this.maxLen) {
      ids.push(this.padId);
    }
    return ids;
  }
}
