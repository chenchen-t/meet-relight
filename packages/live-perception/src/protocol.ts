export type WorkerRequest =
  | { type: 'init'; base: string }
  | { type: 'frame'; id: number; bitmap: ImageBitmap; timestampMs: number; segment: boolean };

export type WorkerResponse =
  | { type: 'ready' }
  | { type: 'error'; message: string }
  | {
      type: 'result';
      id: number;
      landmarks: Float32Array | null;
      matrix: Float32Array | null;
      mask: Uint8Array | null;
      maskWidth: number;
      maskHeight: number;
      faceMs: number;
      segMs: number;
    };

export type PerceptionResult = Extract<WorkerResponse, { type: 'result' }>;
