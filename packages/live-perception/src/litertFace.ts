import { loadAndCompile, Tensor, type CompiledModel } from '@litertjs/core';
import warpSrc from './litertFace.wgsl?raw';

export interface FaceRoi {
  cx: number; // frame pixels
  cy: number;
  size: number;
  angle: number; // radians, y-down image rotation of the face's eye line
}

export interface LiteRtFaceResult {
  /** 478 × (x/W, y/H, z/W) — same convention as MediaPipe normalized landmarks. */
  landmarks: Float32Array;
  presence: number;
  roi: FaceRoi;
}

const LANDMARKS = 478;
const LM_SIZE = 256;
const ROI_SCALE = 1.5; // MediaPipe face landmarker uses a 1.5× square ROI
const PRESENCE_MIN = 0.5;
const DET_MIN = 0.5;

/**
 * Face landmarks on WebGPU via LiteRT.js (MediaPipe face_detector + face_landmarks_detector),
 * with pre/post-processing graphs authored in C++ with the Tensor API (native/tensor_graphs):
 *
 *   detect (only when not tracking):  frame ─face_det_pre→ ─face_detector→ ─face_det_post→ det[17]
 *   track (every frame):  camera ─ROI warp (WGSL)→ ─face_landmarks_detector→ ─face_lm_post(roi)→ face[1435]
 *
 * Only det[17] (68 B, detection frames) and face[1435] (5.7 KB) are read back.
 * Next ROI comes from the landmarks (no detector while tracking).
 */
export class LiteRtFace {
  roi: FaceRoi | null = null;
  busy = false;
  timings = { detMs: 0, lmMs: 0 };
  detections = 0;
  private warpPipe: GPUComputePipeline;
  private warpU: GPUBuffer;
  private sampler: GPUSampler;
  private lmInBuf: GPUBuffer;
  private lmInTensor: Tensor;
  private roiBuf: GPUBuffer;
  private roiTensor: Tensor;

  private constructor(
    private device: GPUDevice,
    private detPre: CompiledModel,
    private det: CompiledModel,
    private detPost: CompiledModel,
    private lm: CompiledModel,
    private lmPost: CompiledModel,
    readonly frameW: number,
    readonly frameH: number,
  ) {
    const module = device.createShaderModule({ label: 'face-warp', code: warpSrc });
    this.warpPipe = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'warp' } });
    this.warpU = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.lmInBuf = device.createBuffer({ label: 'face-lm-input', size: LM_SIZE * LM_SIZE * 3 * 4, usage: storage });
    this.lmInTensor = new Tensor(this.lmInBuf, [1, LM_SIZE, LM_SIZE, 3], 'float32');
    this.roiBuf = device.createBuffer({ label: 'face-roi', size: 16, usage: storage });
    this.roiTensor = new Tensor(this.roiBuf, [1, 4], 'float32');
  }

  /** Requires loadLiteRt() + setWebGpuDevice() to have been called (LiteRtSegmenter.create does both). */
  static async create(device: GPUDevice, base = '/', frameW = 512, frameH = 288): Promise<LiteRtFace> {
    const opts = { accelerator: 'webgpu' as const };
    const m = (p: string) => loadAndCompile(`${base}models/${p}`, opts);
    const [detPre, det, detPost, lm, lmPost] = await Promise.all([
      m('face_det_pre.tflite'),
      m('face_landmarker/face_detector.tflite'),
      m('face_det_post.tflite'),
      m('face_landmarker/face_landmarks_detector_b1.tflite'),
      m('face_lm_post.tflite'),
    ]);
    return new LiteRtFace(device, detPre, det, detPost, lm, lmPost, frameW, frameH);
  }

  /**
   * One tracking step. `camera` must be the frame being displayed; `frameTensor`
   * the packed [1,H,W,4] frame (from LiteRtSegmenter) used only for detection.
   * Resolves with landmarks, or null if no face / this step was a detection.
   */
  async run(camera: GPUExternalTexture, frameTensor: Tensor): Promise<LiteRtFaceResult | null> {
    if (this.busy) return null;
    this.busy = true;
    try {
      if (!this.roi) {
        const t0 = performance.now();
        this.roi = await this.detect(frameTensor);
        this.timings.detMs = performance.now() - t0;
        this.detections++;
        return null; // camera texture expired across the await; landmarks start next frame
      }
      const t0 = performance.now();
      const roi = this.roi;
      // Warp must be recorded synchronously (external textures expire at the end of the task).
      this.device.queue.writeBuffer(this.roiBuf, 0, new Float32Array([roi.cx, roi.cy, roi.size, roi.angle]));
      this.device.queue.writeBuffer(this.warpU, 0, new Float32Array([roi.cx, roi.cy, roi.size, roi.angle, this.frameW, this.frameH, LM_SIZE, 0]));
      const enc = this.device.createCommandEncoder({ label: 'face-warp' });
      const cp = enc.beginComputePass();
      cp.setPipeline(this.warpPipe);
      cp.setBindGroup(0, this.device.createBindGroup({
        layout: this.warpPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: this.sampler },
          { binding: 1, resource: camera },
          { binding: 2, resource: { buffer: this.lmInBuf } },
          { binding: 3, resource: { buffer: this.warpU } },
        ],
      }));
      cp.dispatchWorkgroups(LM_SIZE / 8, LM_SIZE / 8);
      cp.end();
      this.device.queue.submit([enc.finish()]);

      const outs = await this.lm.run(this.lmInTensor);
      const [landmarksT, presenceT] = outs;
      const post = await this.lmPost.run({ landmarks: landmarksT!, presence: presenceT!, roi: this.roiTensor });
      for (const t of outs) t.delete();
      const faceT = post['face']!;
      const face = (await faceT.data()) as Float32Array;
      faceT.delete();
      this.timings.lmMs = performance.now() - t0;

      const presence = face[LANDMARKS * 3]!;
      if (presence < PRESENCE_MIN) {
        this.roi = null;
        return null;
      }
      const landmarks = face.slice(0, LANDMARKS * 3);
      this.roi = this.roiFromLandmarks(landmarks);
      return { landmarks, presence, roi };
    } finally {
      this.busy = false;
    }
  }

  private async detect(frameTensor: Tensor): Promise<FaceRoi | null> {
    const [input] = await this.detPre.run(frameTensor);
    const outs = await this.det.run(input!);
    input!.delete();
    const post = await this.detPost.run({ regressors: outs[0]!, classificators: outs[1]! });
    for (const t of outs) t.delete();
    const detT = post['det']!;
    const d = (await detT.data()) as Float32Array;
    detT.delete();
    if (d[16]! < DET_MIN) return null;
    // Letterboxed square (side = W, vertical pad (W−H)/2) → frame pixels.
    const side = this.frameW;
    const pad = (this.frameW - this.frameH) / 2;
    const px = (x: number) => x * side;
    const py = (y: number) => y * side - pad;
    const w = d[2]! * side;
    const h = d[3]! * side;
    // Keypoints 0 / 1 = right / left eye (subject's), i.e. image-left / image-right.
    const angle = Math.atan2(py(d[7]!) - py(d[5]!), px(d[6]!) - px(d[4]!));
    return { cx: px(d[0]!), cy: py(d[1]!), size: Math.max(w, h) * ROI_SCALE, angle };
  }

  /** Square ROI around the landmarks in the face-aligned frame (MediaPipe FaceLandmarksToRoi). */
  private roiFromLandmarks(lm: Float32Array): FaceRoi {
    const W = this.frameW;
    const H = this.frameH;
    const x = (i: number) => lm[i * 3]! * W;
    const y = (i: number) => lm[i * 3 + 1]! * H;
    const angle = Math.atan2(y(263) - y(33), x(263) - x(33)); // eye outer corners
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    let minU = Infinity, maxU = -Infinity, minV = Infinity, maxV = -Infinity;
    for (let i = 0; i < LANDMARKS; i++) {
      const u = c * x(i) + s * y(i);
      const v = -s * x(i) + c * y(i);
      minU = Math.min(minU, u); maxU = Math.max(maxU, u);
      minV = Math.min(minV, v); maxV = Math.max(maxV, v);
    }
    const cu = (minU + maxU) / 2;
    const cv = (minV + maxV) / 2;
    return {
      cx: c * cu - s * cv,
      cy: s * cu + c * cv,
      size: Math.max(maxU - minU, maxV - minV) * ROI_SCALE,
      angle,
    };
  }
}
