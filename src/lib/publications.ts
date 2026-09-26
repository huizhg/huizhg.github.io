import { getCollection, type CollectionEntry } from 'astro:content';

export type Publication = CollectionEntry<'publications'>;

/** All publications, newest year first, keeping the order of publications.bib within a year. */
export async function getPublications(): Promise<Publication[]> {
  const pubs = await getCollection('publications');
  return pubs.sort((a, b) => b.data.year - a.data.year || a.data.order - b.data.order);
}

const escapeHtml = (text: string) =>
  text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

/** Author list as HTML, with **name** turned into <strong>name</strong>. */
export function authorsHtml(authors: string): string {
  return escapeHtml(authors).replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
}
