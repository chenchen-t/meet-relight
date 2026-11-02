import geoSrc from './geo.wgsl?raw';
import { FACE_TRIANGLES, FACE_VERTEX_COUNT } from './faceMeshData.ts';
import { FACE_VERTEX_STRIDE, type HeadProxy } from './geometry.ts';

export interface GeoOptions {
  /** Silhouette normal tilt strength. */
  blobStrength: number;
  /** Person-mask blur radius in mask texels. */
  blurRadius: number;
  /** Face-mesh → head-proxy feather radius in geo pixels (wide = no visible "mask" edge). */
  faceFeatherPx: number;
  /** Head proxy front flattening (>1 = flatter than a sphere). */
  headFlatten: number;
  /** Relight the seg "hair" class too (off = hair keeps its original lighting). */
  relightHair: boolean;
  /** Softness of the relight coverage boundary (silhouette, hairline, ears) in geo pixels. */
  edgeSoftPx: number;
  /** Gaussian radius (geo px) for smoothing the normal field: softens nose / jawline highlights. 0 = off. */
  normalSmoothPx: number;
}

export const DEFAULT_GEO_OPTIONS: GeoOptions = { blobStrength: 18, blurRadius: 10, faceFeatherPx: 14, headFlatten: 1.25, relightHair: false, edgeSoftPx: 8, normalSmoothPx: 5 };

/**
 * Builds the frozen texture contract (u_geo_tex, u_mask_tex, u_face_uv_tex)
 * from perception results, entirely on the GPU except for a 468-vertex upload.
 *
 * Normal field = face mesh (feathered) over a head-proxy ellipsoid (forehead,
 * temples, ears, hair) over silhouette "blob" normals (neck, shoulders, body).
 */
export class GeoBuilder {
  geo!: GPUTexture; // rgba16float: normal xyz, w = raw face-mesh coverage
  faceUv!: GPUTexture; // rgba16float: canonical uv, b = valid
  mask!: GPUTexture; // rgba8unorm (seg mask, uploaded)
  private faceGeo!: GPUTexture;
  private depth!: GPUTexture;
  private faceBlurA!: GPUTexture;
  private faceBlurB!: GPUTexture;
  private blurA!: GPUTexture;
  private blurB!: GPUTexture;
  private headBlurA!: GPUTexture;
  private headBlurB!: GPUTexture;
  private geoRaw!: GPUTexture;
  private geoTmp!: GPUTexture;

  private facePipe: GPURenderPipeline;
  private blurPipe: GPUComputePipeline;
  private smoothPipe: GPUComputePipeline;
  private composePipe: GPUComputePipeline;
  private vbuf: GPUBuffer;
  private ibuf: GPUBuffer;
  private blurU: GPUBuffer[];
  private composeU: GPUBuffer;
  private sampler: GPUSampler;
  hasFace = false;
  head: HeadProxy | null = null;
  /** uv shift (current − perception-frame head position) applied to the seg mask around the head. */
  maskShift: [number, number] = [0, 0];
  opts: GeoOptions = { ...DEFAULT_GEO_OPTIONS };

  constructor(
    private device: GPUDevice,
    public width: number,
    public height: number,
  ) {
    const module = device.createShaderModule({ label: 'geo', code: geoSrc });
    this.facePipe = device.createRenderPipeline({
      label: 'face-raster',
      layout: 'auto',
      vertex: {
        module,
        entryPoint: 'vsFace',
        buffers: [
          {
            arrayStride: FACE_VERTEX_STRIDE * 4,
            attributes: [
              { shaderLocation: 0, offset: 0, format: 'float32x3' },
              { shaderLocation: 1, offset: 12, format: 'float32x3' },
              { shaderLocation: 2, offset: 24, format: 'float32x2' },
            ],
          },
        ],
      },
      fragment: { module, entryPoint: 'fsFace', targets: [{ format: 'rgba16float' }, { format: 'rgba16float' }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    });
    this.blurPipe = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'blur' } });
    this.smoothPipe = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'smoothN' } });
    this.composePipe = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'compose' } });
    this.vbuf = device.createBuffer({
      size: FACE_VERTEX_COUNT * FACE_VERTEX_STRIDE * 4,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    this.ibuf = device.createBuffer({ size: FACE_TRIANGLES.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.ibuf, 0, FACE_TRIANGLES);
    const ub = (size: number) => device.createBuffer({ size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    // person H/V, face-mesh H/V, face-skin H/V, normal-smooth H/V
    this.blurU = [ub(16), ub(16), ub(16), ub(16), ub(16), ub(16), ub(16), ub(16)];
    this.composeU = ub(64);
    this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    this.allocate();
    this.setMask(new Uint8Array(4), 1, 1);
  }

  private allocate(): void {
    const d = this.device;
    const sz = [this.width, this.height];
    const tex = (format: GPUTextureFormat, usage: number) => d.createTexture({ size: sz, format, usage });
    const RT = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
    const ST = GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING;
    this.faceGeo = tex('rgba16float', RT);
    this.faceUv = tex('rgba16float', RT);
    this.depth = tex('depth24plus', GPUTextureUsage.RENDER_ATTACHMENT);
    this.geo = tex('rgba16float', ST);
    this.geoRaw = tex('rgba16float', ST);
    this.geoTmp = tex('rgba16float', ST);
    this.faceBlurA = tex('rgba16float', ST);
    this.faceBlurB = tex('rgba16float', ST);
  }

  resize(w: number, h: number): void {
    if (w === this.width && h === this.height) return;
    for (const t of [this.faceGeo, this.faceUv, this.depth, this.geo, this.geoRaw, this.geoTmp, this.faceBlurA, this.faceBlurB]) t.destroy();
    this.width = w;
    this.height = h;
    this.allocate();
  }

  setMask(data: Uint8Array, w: number, h: number): void {
    if (!this.mask || this.externalMask || this.mask.width !== w || this.mask.height !== h) {
      if (!this.externalMask) this.mask?.destroy();
      this.externalMask = false;
      this.allocBlur(w, h);
      this.mask = this.device.createTexture({
        size: [w, h],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
    }
    this.device.queue.writeTexture({ texture: this.mask }, data as Uint8Array<ArrayBuffer>, { bytesPerRow: w * 4 }, [w, h]);
  }

  /** Use a GPU-produced mask (e.g. LiteRT.js segmentation) directly — no upload. Not owned. */
  useExternalMask(texture: GPUTexture): void {
    if (this.mask === texture) return;
    if (!this.externalMask) this.mask?.destroy();
    this.externalMask = true;
    this.mask = texture;
    this.allocBlur(texture.width, texture.height);
  }

  private externalMask = false;

  private allocBlur(w: number, h: number): void {
    if (this.blurA && this.blurA.width === w && this.blurA.height === h) return;
    this.blurA?.destroy();
    this.blurB?.destroy();
    this.headBlurA?.destroy();
    this.headBlurB?.destroy();
    const bt = () =>
      this.device.createTexture({
        size: [w, h],
        format: 'rgba16float',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      });
    this.blurA = bt();
    this.blurB = bt();
    this.headBlurA = bt();
    this.headBlurB = bt();
  }

  setFaceVertices(vertices: Float32Array | null): void {
    this.hasFace = !!vertices;
    if (vertices) this.device.queue.writeBuffer(this.vbuf, 0, vertices as Float32Array<ArrayBuffer>);
  }

  setHead(head: HeadProxy | null): void {
    this.head = head;
  }

  private blurPass(cp: GPUComputePassEncoder, src: GPUTexture, dst: GPUTexture, u: GPUBuffer): void {
    cp.setBindGroup(0, this.device.createBindGroup({
      layout: this.blurPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: src.createView() },
        { binding: 1, resource: dst.createView() },
        { binding: 2, resource: { buffer: u } },
      ],
    }));
    cp.dispatchWorkgroups(Math.ceil(src.width / 8), Math.ceil(src.height / 8));
  }

  /** Records face raster + blurs + compose into `encoder`. */
  encode(encoder: GPUCommandEncoder): void {
    const d = this.device;
    const rp = encoder.beginRenderPass({
      colorAttachments: [
        { view: this.faceGeo.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] },
        { view: this.faceUv.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] },
      ],
      depthStencilAttachment: { view: this.depth.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'discard' },
    });
    if (this.hasFace) {
      rp.setPipeline(this.facePipe);
      rp.setVertexBuffer(0, this.vbuf);
      rp.setIndexBuffer(this.ibuf, 'uint16');
      rp.drawIndexed(FACE_TRIANGLES.length);
    }
    rp.end();

    const r = this.opts.blurRadius;
    const fr = Math.max(1, Math.round(this.opts.faceFeatherPx));
    d.queue.writeBuffer(this.blurU[0]!, 0, new Int32Array([1, 0, r, 0]));
    d.queue.writeBuffer(this.blurU[1]!, 0, new Int32Array([0, 1, r, 0]));
    d.queue.writeBuffer(this.blurU[2]!, 0, new Int32Array([1, 0, fr, 3])); // face coverage lives in .w
    d.queue.writeBuffer(this.blurU[3]!, 0, new Int32Array([0, 1, fr, 0]));
    d.queue.writeBuffer(this.blurU[4]!, 0, new Int32Array([1, 0, r, 2])); // seg face-skin lives in .b
    d.queue.writeBuffer(this.blurU[5]!, 0, new Int32Array([0, 1, r, 0]));
    const h = this.head;
    d.queue.writeBuffer(this.composeU, 0, new Float32Array([
      this.opts.blobStrength, h ? 1 : 0, this.opts.headFlatten, this.opts.relightHair ? 1 : 0,
      h?.cx ?? 0.5, h?.cy ?? 0.5, h?.rx ?? 1, h?.ry ?? 1,
      h?.roll ?? 0, h?.yaw ?? 0, this.maskShift[0], this.maskShift[1],
      this.opts.edgeSoftPx, 0, 0, 0,
    ]));

    const cp = encoder.beginComputePass();
    cp.setPipeline(this.blurPipe);
    this.blurPass(cp, this.mask, this.blurA, this.blurU[0]!);
    this.blurPass(cp, this.blurA, this.blurB, this.blurU[1]!);
    this.blurPass(cp, this.faceGeo, this.faceBlurA, this.blurU[2]!);
    this.blurPass(cp, this.faceBlurA, this.faceBlurB, this.blurU[3]!);
    this.blurPass(cp, this.mask, this.headBlurA, this.blurU[4]!);
    this.blurPass(cp, this.headBlurA, this.headBlurB, this.blurU[5]!);
    cp.setPipeline(this.composePipe);
    cp.setBindGroup(0, d.createBindGroup({
      layout: this.composePipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.sampler },
        { binding: 1, resource: this.mask.createView() },
        { binding: 2, resource: this.blurB.createView() },
        { binding: 3, resource: this.faceGeo.createView() },
        { binding: 4, resource: this.geoRaw.createView() },
        { binding: 5, resource: { buffer: this.composeU } },
        { binding: 6, resource: this.faceBlurB.createView() },
        { binding: 7, resource: this.headBlurB.createView() },
      ],
    }));
    cp.dispatchWorkgroups(Math.ceil(this.width / 8), Math.ceil(this.height / 8));
    const sr = Math.max(0, Math.round(this.opts.normalSmoothPx));
    d.queue.writeBuffer(this.blurU[6]!, 0, new Int32Array([1, 0, sr, 0]));
    d.queue.writeBuffer(this.blurU[7]!, 0, new Int32Array([0, 1, sr, 0]));
    cp.setPipeline(this.smoothPipe);
    for (const [src, dst, u] of [[this.geoRaw, this.geoTmp, this.blurU[6]!], [this.geoTmp, this.geo, this.blurU[7]!]] as const) {
      cp.setBindGroup(0, d.createBindGroup({
        layout: this.smoothPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: src.createView() },
          { binding: 1, resource: dst.createView() },
          { binding: 2, resource: { buffer: u } },
        ],
      }));
      cp.dispatchWorkgroups(Math.ceil(this.width / 8), Math.ceil(this.height / 8));
    }
    cp.end();
  }
}
