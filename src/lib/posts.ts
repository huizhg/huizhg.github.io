import { getCollection, type CollectionEntry } from 'astro:content';

export type Post = CollectionEntry<'blog'>;

/** All posts, newest first. Drafts show in `npm run dev` and are left out of builds. */
export async function getPosts(): Promise<Post[]> {
  const posts = await getCollection('blog', ({ data }) => import.meta.env.DEV || !data.draft);
  return posts.sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf());
}

const WORDS_PER_MINUTE = 220;

export function readingTime(body = ''): number {
  const words = body
    .replace(/^---[\s\S]*?---/, ' ')
    .replace(/<[^>]+>/g, ' ')
    .split(/\s+/)
    .filter((w) => /[\p{L}\p{N}]/u.test(w)).length;
  return Math.max(1, Math.ceil(words / WORDS_PER_MINUTE));
}

export function formatDate(date: Date): string {
  return date.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    timeZone: 'UTC',
  });
}

export function groupByYear<T>(items: T[], yearOf: (item: T) => number): [number, T[]][] {
  const groups = new Map<number, T[]>();
  for (const item of items) {
    const year = yearOf(item);
    groups.set(year, [...(groups.get(year) ?? []), item]);
  }
  return [...groups.entries()].sort((a, b) => b[0] - a[0]);
}
