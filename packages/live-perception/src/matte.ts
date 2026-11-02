import matteSrc from './matte.wgsl?raw';

export interface MatteOptions {
  /** Box radius of the guided filter, in matte pixels. */
  radius: number;
  /** Guided-filter regulariser ε (I in [0,1]); smaller = follows image edges more tightly. */
  eps: number;
  /** Temporal blend for still pixels (1 = off). Moving pixels always take the new value. */
  temporal: number;
}

export const DEFAULT_MATTE_OPTIONS: MatteOptions = { radius: 6, eps: 2e-3, temporal: 0.45 };

/**
 * Refines the 256² segmentation person channel to a full-detail matte at `width × height`
 * (fast guided filter guided by the camera frame) and stabilises it over time. ~0.3 ms.
 * `matte` (r16-ish in rgba16float .r) is the output; it is in image (unmirrored) space like the mask.
 */
export class MatteRefiner {
  private packPipe: GPUComputePipeline;
  private boxPipe: GPUComputePipeline;
  private coefPipe: GPUComputePipeline;
  private applyPipe: GPUComputePipeline;
  private sampler: GPUSampler;
  private packed!: GPUTexture;
  private tmp!: GPUTexture;
  private means!: GPUTexture;
  private ab!: GPUTexture;
  private abMean!: GPUTexture;
  private out: GPUTexture[] = [];
  private cur = 0;
  private boxU: GPUBuffer[];
  private paramU: GPUBuffer;
  opts: MatteOptions = { ...DEFAULT_MATTE_OPTIONS };

  constructor(private device: GPUDevice, public width: number, public height: number) {
    const module = device.createShaderModule({ label: 'matte', code: matteSrc });
    const cp = (entryPoint: string) => device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint } });
    this.packPipe = cp('pack');
    this.boxPipe = cp('box');
    this.coefPipe = cp('coef');
    this.applyPipe = cp('apply');
    this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    const ub = () => device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.boxU = [ub(), ub()];
    this.paramU = ub();
    this.allocate();
  }

  /** Latest refined matte (.r). */
  get matte(): GPUTexture {
    return this.out[this.cur]!;
  }

  private allocate(): void {
    const t = () => this.device.createTexture({
      size: [this.width, this.height],
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.packed = t();
    this.tmp = t();
    this.means = t();
    this.ab = t();
    this.abMean = t();
    this.out = [t(), t()];
  }

  private dispatch(pass: GPUComputePassEncoder, pipe: GPUComputePipeline, entries: GPUBindGroupEntry[]): void {
    pass.setPipeline(pipe);
    pass.setBindGroup(0, this.device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries }));
    pass.dispatchWorkgroups(Math.ceil(this.width / 8), Math.ceil(this.height / 8));
  }

  private boxPass(pass: GPUComputePassEncoder, src: GPUTexture, dst: GPUTexture): void {
    this.dispatch(pass, this.boxPipe, [
      { binding: 0, resource: src.createView() },
      { binding: 1, resource: this.tmp.createView() },
      { binding: 2, resource: { buffer: this.boxU[0]! } },
    ]);
    this.dispatch(pass, this.boxPipe, [
      { binding: 0, resource: this.tmp.createView() },
      { binding: 1, resource: dst.createView() },
      { binding: 2, resource: { buffer: this.boxU[1]! } },
    ]);
  }

  /** Record refinement of `mask` (seg, .r = person) guided by `camera`. Call before any await (external texture). */
  encode(encoder: GPUCommandEncoder, camera: GPUExternalTexture, mask: GPUTexture): void {
    const d = this.device;
    const r = Math.max(1, Math.round(this.opts.radius));
    d.queue.writeBuffer(this.boxU[0]!, 0, new Int32Array([1, 0, r, 0]));
    d.queue.writeBuffer(this.boxU[1]!, 0, new Int32Array([0, 1, r, 0]));
    d.queue.writeBuffer(this.paramU, 0, new Float32Array([this.opts.eps, this.opts.temporal, 0, 0]));
    const prev = this.out[this.cur]!;
    this.cur ^= 1;
    const next = this.out[this.cur]!;
    const pass = encoder.beginComputePass({ label: 'matte-refine' });
    this.dispatch(pass, this.packPipe, [
      { binding: 0, resource: this.sampler },
      { binding: 1, resource: camera },
      { binding: 2, resource: mask.createView() },
      { binding: 3, resource: this.packed.createView() },
    ]);
    this.boxPass(pass, this.packed, this.means);
    this.dispatch(pass, this.coefPipe, [
      { binding: 0, resource: this.means.createView() },
      { binding: 1, resource: this.ab.createView() },
      { binding: 2, resource: { buffer: this.paramU } },
    ]);
    this.boxPass(pass, this.ab, this.abMean);
    this.dispatch(pass, this.applyPipe, [
      { binding: 0, resource: this.abMean.createView() },
      { binding: 1, resource: this.packed.createView() },
      { binding: 2, resource: prev.createView() },
      { binding: 3, resource: next.createView() },
      { binding: 4, resource: { buffer: this.paramU } },
    ]);
    pass.end();
  }
}
