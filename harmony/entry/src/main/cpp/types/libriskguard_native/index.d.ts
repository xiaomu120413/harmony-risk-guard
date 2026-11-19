export const loadModel: (path: string) => number;
export const loadModelFromBuffer: (buffer: ArrayBuffer | Uint8Array) => number;
export const diagnose: (buffer: ArrayBuffer | Uint8Array, path: string) => number[];
export const predict: (ids: number[], mask: number[]) => number[];
