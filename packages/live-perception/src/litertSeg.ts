import { loadLiteRt, loadAndCompile, setWebGpuDevice, Tensor, type CompiledModel } from '@litertjs/core';
import glueSrc from './litertSeg.wgsl?raw';

export interface LiteRtSegTimings {
  /** CPU time spent issuing the 3 LiteRT runs (pre, core, post). */
  issueMs: number;
  /** Submit → GPU work done for the whole chain (pack → unpack), measured async. */
  gpuMs: number;
}

/**
 * Selfie segmentation fully on WebGPU via LiteRT.js, sharing the app's GPUDevice:
 *
 *   camera ─pack(WGSL)→ frame GPUBuffer ─seg_pre.tflite→ ─selfie_multiclass→ ─seg_post.tflite→ mask GPUBuffer
 *          ─unpack(WGSL)→ rgba8unorm mask texture (soft person / skin / face / hair)
 *
 * seg_pre / seg_post are authored in C++ with the LiteRT Tensor API
 * (native/tensor_graphs). No CPU readback anywhere.
 */
export class LiteRtSegmenter {
  readonly mask: GPUTexture;
  timings: LiteRtSegTimings = { issueMs: 0, gpuMs: 0 };
  busy = false;
  private frameBuf: GPUBuffer;
  /** Packed camera frame [1,H,W,4] (refreshed by each run(); reused by face detection). */
  readonly frameTensor: Tensor;
  private packPipe: GPUComputePipeline;
  private unpackPipe: GPUComputePipeline;
  private packU: GPUBuffer;
  private sampler: GPUSampler;

  private constructor(
    private device: GPUDevice,
    private pre: CompiledModel,
    private core: CompiledModel,
    private post: CompiledModel,
    readonly frameW: number,
    readonly frameH: number,
  ) {
    const module = device.createShaderModule({ label: 'litert-seg-glue', code: glueSrc });
    this.packPipe = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'pack' } });
    this.unpackPipe = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'unpack' } });
    this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    this.packU = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.packU, 0, new Uint32Array([frameW, frameH, 0, 0]));
    this.frameBuf = device.createBuffer({
      label: 'seg-frame',
      size: frameW * frameH * 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.frameTensor = new Tensor(this.frameBuf, [1, frameH, frameW, 4], 'float32');
    this.mask = device.createTexture({
      label: 'litert-seg-mask',
      size: [256, 256],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });
  }

  static async create(device: GPUDevice, base = '/', frameW = 512, frameH = 288): Promise<LiteRtSegmenter> {
    await loadLiteRt(`${base}litert/wasm/`);
    setWebGpuDevice(device); // LiteRT.js shares our device → tensors are plain GPUBuffers on it
    const opts = { accelerator: 'webgpu' as const };
    const [pre, core, post] = await Promise.all([
      loadAndCompile(`${base}models/seg_pre.tflite`, opts),
      loadAndCompile(`${base}models/selfie_multiclass_256x256.tflite`, opts),
      loadAndCompile(`${base}models/seg_post.tflite`, opts),
    ]);
    return new LiteRtSegmenter(device, pre, core, post, frameW, frameH);
  }

  describe(): string {
    const fmt = (m: CompiledModel) =>
      `in ${m.getInputDetails().map((d) => `${d.name}[${[...d.shape]}]`).join(',')} → out ${m
        .getOutputDetails()
        .map((d) => `${d.name}[${[...d.shape]}]`)
        .join(',')}`;
    return [`pre:  ${fmt(this.pre)}`, `core: ${fmt(this.core)}`, `post: ${fmt(this.post)}`].join('\n');
  }

  /**
   * Runs the chain on `camera`. Resolves once all GPU work has been submitted
   * (the mask texture is valid for any work submitted afterwards on the same queue).
   */
  async run(camera: GPUExternalTexture): Promise<void> {
    if (this.busy) return;
    this.busy = true;
    const d = this.device;
    try {
      const t0 = performance.now();
      const enc = d.createCommandEncoder({ label: 'seg-pack' });
      const cp = enc.beginComputePass();
      cp.setPipeline(this.packPipe);
      cp.setBindGroup(0, d.createBindGroup({
        layout: this.packPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: this.sampler },
          { binding: 1, resource: camera },
          { binding: 2, resource: { buffer: this.frameBuf } },
          { binding: 3, resource: { buffer: this.packU } },
        ],
      }));
      cp.dispatchWorkgroups(Math.ceil(this.frameW / 8), Math.ceil(this.frameH / 8));
      cp.end();
      d.queue.submit([enc.finish()]);

      const [normalized] = await this.pre.run(this.frameTensor);
      const [logits] = await this.core.run(normalized!);
      normalized!.delete();
      const [mask] = await this.post.run(logits!);
      logits!.delete();
      const t1 = performance.now();

      const enc2 = d.createCommandEncoder({ label: 'seg-unpack' });
      const cp2 = enc2.beginComputePass();
      cp2.setPipeline(this.unpackPipe);
      cp2.setBindGroup(0, d.createBindGroup({
        layout: this.unpackPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: mask!.toGpuBuffer() } },
          { binding: 1, resource: this.mask.createView() },
        ],
      }));
      cp2.dispatchWorkgroups(32, 32);
      cp2.end();
      d.queue.submit([enc2.finish()]);
      mask!.delete();
      this.timings.issueMs = t1 - t0;
      void d.queue.onSubmittedWorkDone().then(() => (this.timings.gpuMs = performance.now() - t0));
    } finally {
      this.busy = false;
    }
  }
}
