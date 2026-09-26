import { site } from '../../content/site';
import { formatEntry, latexToText, parseAuthors, parseBibtex } from './bibtex';

/** Extra fields that control the site. They are left out of the BibTeX visitors copy. */
const SITE_FIELDS = ['selected', 'venue', 'pdf', 'code'];

const normalizeName = (name: string) =>
  name
    .normalize('NFD')
    .replace(/[̀-ͯ]/g, '')
    .toLowerCase()
    .replace(/[^a-z]+/g, ' ')
    .trim();

const myNames = new Set(site.authorNames.map(normalizeName));

/** Parser for content/publications.bib, used by the publications collection. */
export function publicationsFromBibtex(text: string) {
  return parseBibtex(text).flatMap((entry, order) => {
    const f = entry.fields;
    const year = parseInt(f.year ?? '', 10);
    if (!f.title || Number.isNaN(year)) {
      console.warn(`[publications.bib] Skipped "${entry.key}": every entry needs a title and a year.`);
      return [];
    }
    const authors = parseAuthors(f.author ?? '')
      .map((name) => (myNames.has(normalizeName(name)) ? `**${name}**` : name))
      .join(', ');
    const venue = f.venue ?? f.journal ?? f.booktitle ?? f.howpublished ?? f.school ?? f.institution ?? f.publisher ?? '';
    return [
      {
        id: entry.key,
        order,
        title: latexToText(f.title),
        authors,
        venue: latexToText(venue),
        year,
        selected: /^(true|yes|1)$/i.test((f.selected ?? '').trim()),
        links: { pdf: (f.pdf ?? '').trim(), code: (f.code ?? '').trim() },
        bibtex: formatEntry(entry, SITE_FIELDS),
      },
    ];
  });
}
