// @ts-check
import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';
import { unified } from '@astrojs/markdown-remark';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeFigure from './src/lib/rehype-figure.mjs';
import { codeLight, codeDark } from './src/styles/code-themes.mjs';

export default defineConfig({
  site: 'https://huizhg.github.io',
  integrations: [sitemap()],
  devToolbar: { enabled: false },
  markdown: {
    processor: unified({
      remarkPlugins: [remarkMath],
      rehypePlugins: [rehypeKatex, rehypeFigure],
    }),
    shikiConfig: {
      themes: { light: codeLight, dark: codeDark },
      defaultColor: false,
    },
  },
});
