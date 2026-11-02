import { defineConfig, configDefaults } from 'vitest/config';

export default defineConfig({
  test: {
    // third_party/ holds checked-out upstream sources (LiteRT, litert-samples) with their own tests.
    exclude: [...configDefaults.exclude, 'third_party/**'],
  },
});
