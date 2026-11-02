import { lightDirection } from '@relight/shared-schema';
import { shProject, type SH9 } from './sh.ts';
import { dot, type Vec3 } from './vec.ts';

/** Simple directional-light description of a physical room, for tests and the playground. */
export interface RoomLightSpec {
  ambient: Vec3;
  lights: { azimuth_deg: number; elevation_deg: number; rgb: Vec3 }[];
}

export const ROOM_PRESETS: Record<string, RoomLightSpec> = {
  /** Office ceiling panel: harsh top light → dark eye sockets and under-chin. */
  overhead: {
    ambient: [0.06, 0.06, 0.07],
    lights: [{ azimuth_deg: 0, elevation_deg: 70, rgb: [0.95, 0.97, 1.0] }],
  },
  /** Side window on the viewer's left, cool daylight. */
  window_left: {
    ambient: [0.08, 0.09, 0.11],
    lights: [{ azimuth_deg: -75, elevation_deg: 15, rgb: [0.85, 0.92, 1.05] }],
  },
  /** Screen glow from below + warm desk lamp right. */
  desk_lamp: {
    ambient: [0.05, 0.05, 0.06],
    lights: [
      { azimuth_deg: 0, elevation_deg: -25, rgb: [0.25, 0.3, 0.4] },
      { azimuth_deg: 60, elevation_deg: 20, rgb: [1.0, 0.75, 0.5] },
    ],
  },
  flat: { ambient: [0.75, 0.75, 0.75], lights: [] },
};

export function roomShadingFn(spec: RoomLightSpec): (n: Vec3) => Vec3 {
  const dirs = spec.lights.map((l) => ({ d: lightDirection(l.azimuth_deg, l.elevation_deg), rgb: l.rgb }));
  return (n) => {
    const out: Vec3 = [...spec.ambient];
    for (const l of dirs) {
      const k = Math.max(dot(n, l.d), 0);
      out[0] += l.rgb[0] * k;
      out[1] += l.rgb[1] * k;
      out[2] += l.rgb[2] * k;
    }
    return out;
  };
}

export function roomSH(spec: RoomLightSpec): SH9 {
  return shProject(roomShadingFn(spec));
}
