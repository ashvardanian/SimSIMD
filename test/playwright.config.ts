import { defineConfig } from '@playwright/test';

export default defineConfig({
    testDir: '.',
    testMatch: ['test-browser-runner.mjs', 'test-browser-bundle-runner.mjs'],
    timeout: 120000,
    reporter: process.env.CI ? [['list'], ['html', { open: 'never' }]] : 'list',
    use: {
        baseURL: 'http://localhost:8888',
    },
    webServer: {
        command: 'npx http-server .. -p 8888 -c-1 --silent',
        port: 8888,
        reuseExistingServer: !process.env.CI,
    },
});
