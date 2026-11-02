/** float32 → IEEE half (uint16) for rgba16float uploads (round-to-nearest-even). */
const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);

export function toHalf(v: number): number {
  f32[0] = v;
  const x = u32[0]!;
  const sign = (x >>> 16) & 0x8000;
  let exp = ((x >>> 23) & 0xff) - 127 + 15;
  let mant = x & 0x7fffff;
  if (((x >>> 23) & 0xff) === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0); // inf/nan
  if (exp >= 0x1f) return sign | 0x7c00; // overflow → inf
  if (exp <= 0) {
    if (exp < -10) return sign; // underflow → 0
    mant = (mant | 0x800000) >>> (1 - exp);
    if (mant & 0x1000) mant += 0x2000;
    return sign | (mant >>> 13);
  }
  if (mant & 0x1000) {
    mant += 0x2000;
    if (mant & 0x800000) {
      mant = 0;
      exp += 1;
      if (exp >= 0x1f) return sign | 0x7c00;
    }
  }
  return sign | (exp << 10) | (mant >>> 13);
}

export function toHalfArray(src: Float32Array): Uint16Array<ArrayBuffer> {
  const out = new Uint16Array(src.length);
  for (let i = 0; i < src.length; i++) out[i] = toHalf(src[i]!);
  return out;
}
