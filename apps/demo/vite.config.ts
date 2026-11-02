import { defineConfig } from 'vite';
import { resolve } from 'node:path';

export default defineConfig({
  // Relative asset URLs so the static build can be hosted under any sub-path (e.g. GitHub Pages).
  base: './',
  server: {
    // Listen on all interfaces (IPv4 + IPv6). Default 'localhost' may bind ::1 only,
    // which breaks browsers that resolve localhost → 127.0.0.1.
    host: true,
    port: 5173,
    strictPort: true,
  },
  build: {
    rollupOptions: {
      input: {
        index: resolve(import.meta.dirname, 'index.html'),
        probe: resolve(import.meta.dirname, 'probe.html'),
        playground: resolve(import.meta.dirname, 'playground.html'),
        live: resolve(import.meta.dirname, 'live.html'),
        litert: resolve(import.meta.dirname, 'litert.html'),
        inspect: resolve(import.meta.dirname, 'inspect.html'),
      },
    },
  },
});
