// Where the site reads your content from. You don't need to edit this file:
// add a folder per post to content/blog/ and papers to content/publications.bib.
import { defineCollection } from 'astro:content';
import { glob, file } from 'astro/loaders';
import { z } from 'astro/zod';
import { publicationsFromBibtex } from './lib/publications-bib';

const blog = defineCollection({
  // One folder per post: content/blog/<slug>/index.md, with the post's images next to it.
  // Folders starting with "_" (like _template) are ignored.
  loader: glob({ pattern: '[^_]*/index.md', base: './content/blog' }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    date: z.coerce.date(),
    updated: z.coerce.date().optional(),
    tags: z.array(z.string()).default([]),
    draft: z.boolean().default(false),
    references: z.array(z.string()).default([]),
  }),
});

const publications = defineCollection({
  loader: file('content/publications.bib', { parser: publicationsFromBibtex }),
  schema: z.object({
    order: z.number(),
    title: z.string(),
    authors: z.string(),
    venue: z.string(),
    year: z.number().int(),
    selected: z.boolean(),
    links: z.object({ pdf: z.string(), code: z.string() }),
    bibtex: z.string(),
  }),
});

export const collections = { blog, publications };
