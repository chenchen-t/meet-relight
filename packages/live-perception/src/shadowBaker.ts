import shadowSrc from './shadow.wgsl?raw';
import { CANONICAL_POSITIONS, CANONICAL_UVS, FACE_TRIANGLES, FACE_VERTEX_COUNT } from './faceMeshData.ts';
import { computeVertexNormals } from './geometry.ts';

export const SHADOW_SIZE = 256;
export const SHADOW_LAYER_YAWS = [-30, 0, 30] as const;

/**
 * Bakes 3 canonical-UV key-light visibility maps (one per head-yaw layer) by
 * GPU ray tracing the canonical face mesh. Re-run whenever the key light changes
 * (~ms on desktop GPUs). Output texture binds to RelightRenderer `shadow`.
 */
export class ShadowBaker {
  readonly texture: GPUTexture;
  private posTex: GPUTexture;
  private nrmTex: GPUTexture;
  private rasterPipe: GPURenderPipeline;
  private tracePipe: GPUComputePipeline;
  private vbuf: GPUBuffer;
  private ibuf: GPUBuffer;
  private vertStore: GPUBuffer;
  private triStore: GPUBuffer;
  private ubuf: GPUBuffer;
  private rasterDone = false;

  constructor(private device: GPUDevice) {
    const module = device.createShaderModule({ label: 'shadow', code: shadowSrc });
    this.rasterPipe = device.createRenderPipeline({
      layout: 'auto',
      vertex: {
        module,
        entryPoint: 'vsUv',
        buffers: [
          {
            arrayStride: 32,
            attributes: [
              { shaderLocation: 0, offset: 0, format: 'float32x3' },
              { shaderLocation: 1, offset: 12, format: 'float32x3' },
              { shaderLocation: 2, offset: 24, format: 'float32x2' },
            ],
          },
        ],
      },
      fragment: { module, entryPoint: 'fsUv', targets: [{ format: 'rgba32float' }, { format: 'rgba32float' }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
    });
    this.tracePipe = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'trace' } });

    const N = computeVertexNormals(CANONICAL_POSITIONS);
    const v = new Float32Array(FACE_VERTEX_COUNT * 8);
    for (let i = 0; i < FACE_VERTEX_COUNT; i++) {
      v.set([CANONICAL_POSITIONS[i * 3]!, CANONICAL_POSITIONS[i * 3 + 1]!, CANONICAL_POSITIONS[i * 3 + 2]!], i * 8);
      v.set([N[i * 3]!, N[i * 3 + 1]!, N[i * 3 + 2]!], i * 8 + 3);
      v.set([CANONICAL_UVS[i * 2]!, 1 - CANONICAL_UVS[i * 2 + 1]!], i * 8 + 6); // texture space, v down
    }
    const buf = (data: ArrayBufferView & { byteLength: number }, usage: number) => {
      const b = device.createBuffer({ size: Math.ceil(data.byteLength / 4) * 4, usage: usage | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(b, 0, data as BufferSource);
      return b;
    };
    this.vbuf = buf(v, GPUBufferUsage.VERTEX);
    this.ibuf = buf(FACE_TRIANGLES, GPUBufferUsage.INDEX);
    this.vertStore = buf(CANONICAL_POSITIONS, GPUBufferUsage.STORAGE);
    this.triStore = buf(Uint32Array.from(FACE_TRIANGLES), GPUBufferUsage.STORAGE);
    this.ubuf = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    const rt = () =>
      device.createTexture({
        size: [SHADOW_SIZE, SHADOW_SIZE],
        format: 'rgba32float',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });
    this.posTex = rt();
    this.nrmTex = rt();
    this.texture = device.createTexture({
      label: 'uv-shadow',
      size: [SHADOW_SIZE, SHADOW_SIZE, 3],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
  }

  get view(): GPUTextureView {
    return this.texture.createView({ dimension: '2d-array' });
  }

  /**
   * @param keyDir key light direction in camera space (unit).
   * @param softness 0..1 → penumbra cone half-angle.
   */
  bake(keyDir: [number, number, number], softness: number, samples = 16): void {
    const d = this.device;
    const enc = d.createCommandEncoder({ label: 'shadow-bake' });
    if (!this.rasterDone) {
      const rp = enc.beginRenderPass({
        colorAttachments: [
          { view: this.posTex.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] },
          { view: this.nrmTex.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] },
        ],
      });
      rp.setPipeline(this.rasterPipe);
      rp.setVertexBuffer(0, this.vbuf);
      rp.setIndexBuffer(this.ibuf, 'uint16');
      rp.drawIndexed(FACE_TRIANGLES.length);
      rp.end();
      this.rasterDone = true;
    }
    const u = new Float32Array(16);
    const [lx, ly, lz] = keyDir;
    SHADOW_LAYER_YAWS.forEach((yaw, k) => {
      // Head yawed by θ: light in head frame = R_y(θ)ᵀ · L.
      const t = (yaw * Math.PI) / 180;
      const c = Math.cos(t);
      const s = Math.sin(t);
      u.set([c * lx - s * lz, ly, s * lx + c * lz, 0], k * 4);
    });
    u[12] = 0.03 + 0.35 * softness; // cone half-angle (rad, small-angle)
    new Uint32Array(u.buffer).set([samples, FACE_TRIANGLES.length / 3, 0], 13);
    d.queue.writeBuffer(this.ubuf, 0, u);
    const cp = enc.beginComputePass();
    cp.setPipeline(this.tracePipe);
    cp.setBindGroup(0, d.createBindGroup({
      layout: this.tracePipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.posTex.createView() },
        { binding: 1, resource: this.nrmTex.createView() },
        { binding: 2, resource: { buffer: this.vertStore } },
        { binding: 3, resource: { buffer: this.triStore } },
        { binding: 4, resource: this.texture.createView({ dimension: '2d-array' }) },
        { binding: 5, resource: { buffer: this.ubuf } },
      ],
    }));
    cp.dispatchWorkgroups(SHADOW_SIZE / 8, SHADOW_SIZE / 8, 3);
    cp.end();
    d.queue.submit([enc.finish()]);
  }
}
