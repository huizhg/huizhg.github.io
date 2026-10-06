// @ts-check
import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';
import { unified } from '@astrojs/markdown-remark';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeFigure from './src/lib/rehype-figure.mjs';
import rehypeCite from './src/lib/rehype-cite.mjs';
import freshDevImages from './src/lib/fresh-dev-images.mjs';
import { codeLight, codeDark } from './src/styles/code-themes.mjs';

export default defineConfig({
  site: 'https://huizhg.github.io',
  integrations: [sitemap(), freshDevImages()],
  devToolbar: { enabled: false },
  markdown: {
    processor: unified({
      remarkPlugins: [remarkMath],
      rehypePlugins: [rehypeKatex, rehypeFigure, rehypeCite],
    }),
    shikiConfig: {
      themes: { light: codeLight, dark: codeDark },
      defaultColor: false,
    },
  },
});
