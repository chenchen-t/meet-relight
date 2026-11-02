import kernelSrc from './relight.wgsl?raw';
import { lightDirection, type TargetProfile, type Light } from '@relight/shared-schema';
import { bakeQuotientMatcap, MATCAP_SIZE, type RoomLight, type BakeOptions } from './matcap.ts';
import { toHalfArray } from './half.ts';

export type SourceKind = 'external' | 'texture';

export enum ViewMode {
  Relit = 0,
  Original = 1,
  Split = 2,
  Normals = 3,
  Ratio = 4,
  Mask = 5,
  Shadow = 6,
}

export interface RelightParams {
  roughness: number;
  specStrength: number;
  rimColor: [number, number, number]; // linear, premultiplied by intensity
  mirror: boolean;
  mode: ViewMode;
  splitX: number;
  strength: number;
  shadowWeights: [number, number, number];
  /** 0..1 how strongly baked cast shadows attenuate the key light. */
  shadowStrength: number;
  /** 0..1 ease relighting toward the original at the silhouette (grazing normals). */
  edgeFade: number;
  /** 0..1 filmic shoulder on added highlights + broader spec lobe (softer nose / jaw highlights). */
  highlightSoft: number;
  /** Show the virtual background set via setBackground() (default true when one is set). */
  background?: boolean;
  /** 0..1 background light wrap over the subject edge. */
  lightWrap?: number;
  /** A refined matte is bound via RelightInputs.matte. */
  refinedMatte?: boolean;
  /** Row-major 3x3 colour-correction matrix (applied in linear space). */
  ccm: number[];
}

export const DEFAULT_PARAMS: RelightParams = {
  roughness: 0.5,
  specStrength: 0.15,
  rimColor: [0, 0, 0],
  mirror: false,
  mode: ViewMode.Relit,
  splitX: 0.5,
  strength: 1,
  shadowWeights: [0, 1, 0],
  shadowStrength: 0.85,
  edgeFade: 0.6,
  highlightSoft: 0.6,
  ccm: [1, 0, 0, 0, 1, 0, 0, 0, 1],
};

export interface RelightInputs {
  source: GPUExternalTexture | GPUTextureView;
  geo: GPUTextureView;
  mask: GPUTextureView;
  /** Optional; defaults to an all-lit 1x1 layer set. */
  shadow?: GPUTextureView;
  /** Optional; defaults to "no face UV" (shadow disabled). */
  faceUv?: GPUTextureView;
  /** Optional refined person matte (.r) for background compositing (set `RelightParams.refinedMatte`). */
  matte?: GPUTextureView;
}

/**
 * Online relight pass (doc §2.3 Stage B). Owns the quotient MatCap texture;
 * call `bake()` whenever the target profile or room light changes.
 */
export class RelightRenderer {
  readonly pipeline: GPURenderPipeline;
  readonly matcap: GPUTexture;
  private sampler: GPUSampler;
  private uniformBuf: GPUBuffer;
  private defaultShadow: GPUTexture;
  private defaultFaceUv: GPUTexture;
  private keyDir: [number, number, number] = [0, 0, 1];
  private keyColor: [number, number, number] = [1, 1, 1];
  private rimDir: [number, number, number] = [0, 0, -1];
  /** Last baked matcap (float32 RGBA), for previews. */
  lastBake: Float32Array | null = null;

  constructor(
    private device: GPUDevice,
    format: GPUTextureFormat,
    readonly sourceKind: SourceKind,
  ) {
    const code = kernelSrc
      .replace(
        '{{SRC_DECL}}',
        sourceKind === 'external'
          ? '@group(0) @binding(1) var srcTex: texture_external;'
          : '@group(0) @binding(1) var srcTex: texture_2d<f32>;',
      )
      .replace(
        '{{SRC_SAMPLE}}',
        sourceKind === 'external'
          ? 'textureSampleBaseClampToEdge(srcTex, samp, uv).rgb'
          : 'textureSampleLevel(srcTex, samp, uv, 0.0).rgb',
      );
    const module = device.createShaderModule({ label: 'relight', code });
    if (code.includes('{{')) throw new Error('relight.wgsl: unresolved template token');
    this.pipeline = device.createRenderPipeline({
      label: 'relight',
      layout: 'auto',
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [{ format }] },
      primitive: { topology: 'triangle-list' },
    });
    this.sampler = device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });
    this.uniformBuf = device.createBuffer({ size: 160, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.matcap = device.createTexture({
      label: 'quotient-matcap',
      size: [MATCAP_SIZE, MATCAP_SIZE],
      format: 'rgba16float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this.defaultShadow = device.createTexture({
      size: [1, 1, 3],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    device.queue.writeTexture(
      { texture: this.defaultShadow },
      new Uint8Array(12).fill(255),
      { bytesPerRow: 4, rowsPerImage: 1 },
      [1, 1, 3],
    );
    this.defaultFaceUv = device.createTexture({
      size: [1, 1],
      format: 'rgba16float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    device.queue.writeTexture({ texture: this.defaultFaceUv }, new Uint16Array(4), { bytesPerRow: 8 }, [1, 1]);
  }

  /** Bake the quotient MatCap for `profile` against `room` and upload it. Returns CPU bake time (ms). */
  bake(profile: TargetProfile, room: RoomLight, opts?: BakeOptions): number {
    const t0 = performance.now();
    const data = bakeQuotientMatcap(profile.lighting_rig, room, {
      ...opts,
      shading: opts?.shading ?? { sss: profile.subject_type === 'portrait' ? 1 : 0, gamma: 1 },
    });
    this.lastBake = data;
    this.device.queue.writeTexture(
      { texture: this.matcap },
      toHalfArray(data),
      { bytesPerRow: MATCAP_SIZE * 8 },
      [MATCAP_SIZE, MATCAP_SIZE],
    );
    const key: Light | undefined =
      profile.lighting_rig.lights.find((l) => l.role === 'key') ?? profile.lighting_rig.lights[0];
    const rim = profile.lighting_rig.lights.find((l) => l.role === 'rim');
    this.rimDir = rim ? lightDirection(rim.azimuth_deg, rim.elevation_deg) : [0, 0, -1];
    if (key) {
      this.keyDir = lightDirection(key.azimuth_deg, key.elevation_deg);
      // color_rgb is treated as a linear light colour everywhere (baker + spec).
      this.keyColor = key.color_rgb.map((c) => c * key.intensity) as [number, number, number];
    } else {
      this.keyColor = [0, 0, 0];
    }
    return performance.now() - t0;
  }

  setParams(p: RelightParams): void {
    const f = new Float32Array(40);
    const c = p.ccm;
    // mat3x3f columns (WGSL is column-major); input is row-major.
    for (let col = 0; col < 3; col++) for (let row = 0; row < 3; row++) f[col * 4 + row] = c[row * 3 + col]!;
    f.set(this.keyDir, 12);
    f[15] = p.roughness;
    f.set(p.rimColor, 16);
    f[19] = p.specStrength;
    f.set(p.shadowWeights, 20);
    f[23] = p.mirror ? 1 : 0;
    f.set(this.keyColor, 24);
    f[27] = p.mode;
    f[28] = p.splitX;
    f[29] = p.strength;
    f[30] = p.shadowStrength;
    f[31] = p.highlightSoft ?? 0;
    f.set(this.rimDir, 32);
    f[35] = p.edgeFade;
    f[36] = this.bgTex && (p.background ?? true) ? 1 : 0;
    f[37] = this.bgAspect;
    f[38] = p.lightWrap ?? 0.35;
    f[39] = p.refinedMatte ? 1 : 0;
    this.device.queue.writeBuffer(this.uniformBuf, 0, f);
  }

  /** Virtual background image (null = keep the camera background). */
  setBackground(image: ImageBitmap | null): void {
    this.bgTex?.destroy();
    this.bgTex = null;
    if (!image) return;
    this.bgTex = this.device.createTexture({
      size: [image.width, image.height],
      format: 'rgba8unorm-srgb',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.device.queue.copyExternalImageToTexture({ source: image }, { texture: this.bgTex }, [image.width, image.height]);
    this.bgAspect = image.width / image.height;
  }
  private bgTex: GPUTexture | null = null;
  private bgAspect = 16 / 9;

  /** Records the relight draw into an already-begun render pass. */
  draw(pass: GPURenderPassEncoder, inputs: RelightInputs): void {
    const bg = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.sampler },
        { binding: 1, resource: inputs.source },
        { binding: 2, resource: inputs.geo },
        { binding: 3, resource: inputs.mask },
        { binding: 4, resource: this.matcap.createView() },
        { binding: 5, resource: inputs.shadow ?? this.defaultShadow.createView({ dimension: '2d-array' }) },
        { binding: 6, resource: inputs.faceUv ?? this.defaultFaceUv.createView() },
        { binding: 7, resource: { buffer: this.uniformBuf } },
        { binding: 8, resource: (this.bgTex ?? this.defaultFaceUv).createView() },
        { binding: 9, resource: inputs.matte ?? this.defaultFaceUv.createView() },
      ],
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bg);
    pass.draw(3);
  }
}
