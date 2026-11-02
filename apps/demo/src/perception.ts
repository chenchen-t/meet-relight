import { FilesetResolver, FaceLandmarker, ImageSegmenter } from '@mediapipe/tasks-vision';

/**
 * M0 perception probe. Runs MediaPipe FaceLandmarker + selfie_multiclass
 * segmenter on the camera frame and returns CPU-side results ready to upload
 * to WebGPU. (MediaPipe Tasks has no WebGPU interop, so masks are read back
 * from its internal WebGL context — that cost is measured as `maskReadMs`.)
 *
 * selfie_multiclass categories: 0 bg, 1 hair, 2 body-skin, 3 face-skin, 4 clothes, 5 others.
 */
export interface PerceptionFrame {
  /** 478 landmarks, normalized image coords (x, y) + relative depth z, flattened xyz. */
  landmarks: Float32Array | null;
  /** 4x4 column-major facial transformation matrix (canonical face → camera), if face found. */
  faceMatrix: Float32Array | null;
  /** RGBA8 mask: r = foreground, g = skin, b = face-skin, a = hair. */
  mask: Uint8Array | null;
  maskWidth: number;
  maskHeight: number;
  timings: { downscaleMs: number; faceMs: number; segMs: number; maskReadMs: number };
}

export class Perception {
  private constructor(
    private face: FaceLandmarker,
    private seg: ImageSegmenter,
  ) {}

  static async create(): Promise<Perception> {
    const fileset = await FilesetResolver.forVisionTasks('./mediapipe/wasm');
    const [face, seg] = await Promise.all([
      FaceLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: './models/face_landmarker.task', delegate: 'GPU' },
        runningMode: 'VIDEO',
        numFaces: 1,
        outputFacialTransformationMatrixes: true,
      }),
      ImageSegmenter.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: './models/selfie_multiclass_256x256.tflite', delegate: 'GPU' },
        runningMode: 'VIDEO',
        outputCategoryMask: true,
        outputConfidenceMasks: false,
      }),
    ]);
    return new Perception(face, seg);
  }

  private rgba = new Uint8Array(0);
  /** Processing resolution (long edge). Masks come back at this size. */
  static PROC_LONG_EDGE = 512;
  private small = document.createElement('canvas');
  private smallCtx = this.small.getContext('2d', { willReadFrequently: false })!;

  private downscale(video: HTMLVideoElement): HTMLCanvasElement {
    const s = Perception.PROC_LONG_EDGE / Math.max(video.videoWidth, video.videoHeight);
    const w = Math.round(video.videoWidth * s);
    const h = Math.round(video.videoHeight * s);
    if (this.small.width !== w || this.small.height !== h) {
      this.small.width = w;
      this.small.height = h;
    }
    this.smallCtx.drawImage(video, 0, 0, w, h);
    return this.small;
  }

  process(video: HTMLVideoElement, timestampMs: number): PerceptionFrame {
    const tA = performance.now();
    const img = this.downscale(video);
    const t0 = performance.now();
    const fr = this.face.detectForVideo(img, timestampMs);
    const t1 = performance.now();

    let landmarks: Float32Array | null = null;
    let faceMatrix: Float32Array | null = null;
    const lm = fr.faceLandmarks[0];
    if (lm) {
      landmarks = new Float32Array(lm.length * 3);
      for (let i = 0; i < lm.length; i++) {
        landmarks[i * 3] = lm[i]!.x;
        landmarks[i * 3 + 1] = lm[i]!.y;
        landmarks[i * 3 + 2] = lm[i]!.z;
      }
      const m = fr.facialTransformationMatrixes?.[0];
      if (m) faceMatrix = Float32Array.from(m.data);
    }

    const sr = this.seg.segmentForVideo(img, timestampMs);
    const t2 = performance.now();
    let mask: Uint8Array | null = null;
    let w = 0;
    let h = 0;
    const cm = sr.categoryMask;
    if (cm) {
      w = cm.width;
      h = cm.height;
      const cat = cm.getAsUint8Array();
      if (this.rgba.length !== w * h * 4) this.rgba = new Uint8Array(w * h * 4);
      const out = this.rgba;
      for (let i = 0, j = 0; i < cat.length; i++, j += 4) {
        const c = cat[i]!;
        out[j] = c !== 0 ? 255 : 0;
        out[j + 1] = c === 2 || c === 3 ? 255 : 0;
        out[j + 2] = c === 3 ? 255 : 0;
        out[j + 3] = c === 1 ? 255 : 0;
      }
      mask = out;
    }
    sr.close();
    const t3 = performance.now();

    return {
      landmarks,
      faceMatrix,
      mask,
      maskWidth: w,
      maskHeight: h,
      timings: { downscaleMs: t0 - tA, faceMs: t1 - t0, segMs: t2 - t1, maskReadMs: t3 - t2 },
    };
  }
}
