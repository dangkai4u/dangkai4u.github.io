import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';

export default defineConfig({
  site: 'https://dangkai4u.github.io',
  output: 'static',
  integrations: [mdx(), sitemap()],
  i18n: {
    defaultLocale: 'zh-cn',
    locales: ['zh-cn', 'en'],
    routing: {
      prefixDefaultLocale: false,
    },
  },
  markdown: {
    shikiConfig: {
      themes: {
        light: 'github-light',
        dark: 'github-dark',
      },
    },
  },
});
