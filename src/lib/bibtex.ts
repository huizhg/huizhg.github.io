// A small BibTeX reader for content/publications.bib.
// Handles what Google Scholar, DBLP and publishers export: braced or quoted values,
// "Last, First" author names, LaTeX accents, and month macros.

export interface BibEntry {
  type: string;
  key: string;
  /** Field values with LaTeX markup still in place, e.g. "{Robust} {3D} pose transfer". */
  fields: Record<string, string>;
  /** Field values exactly as written, including their braces or quotes (used to rebuild the entry). */
  rawFields: [string, string][];
}

const MONTHS: Record<string, string> = {
  jan: 'January', feb: 'February', mar: 'March', apr: 'April', may: 'May', jun: 'June',
  jul: 'July', aug: 'August', sep: 'September', oct: 'October', nov: 'November', dec: 'December',
};

/** Index of the brace or parenthesis that closes the one at `open`, or -1. */
function findClose(text: string, open: number): number {
  const closer = text[open] === '(' ? ')' : '}';
  let depth = 0;
  for (let i = open + 1; i < text.length; i++) {
    const ch = text[i];
    if (ch === '\\') {
      i++;
    } else if (ch === '{') {
      depth++;
    } else if (ch === '}') {
      if (depth === 0 && closer === '}') return i;
      depth--;
    } else if (ch === ')' && depth === 0 && closer === ')') {
      return i;
    }
  }
  return -1;
}

/** Parses `name = value, name = {value} # "more", ...` into raw [name, value] pairs. */
function parseFields(body: string): [string, string][] {
  const out: [string, string][] = [];
  let i = 0;
  // Skips separators and "% comment" lines between fields.
  const skip = () => {
    while (i < body.length) {
      if (/[\s,]/.test(body[i])) i++;
      else if (body[i] === '%') while (i < body.length && body[i] !== '\n') i++;
      else break;
    }
  };
  while (true) {
    skip();
    const name = /^[^\s=,{}"#]+/.exec(body.slice(i));
    if (!name) break;
    i += name[0].length;
    while (/\s/.test(body[i] ?? '')) i++;
    if (body[i] !== '=') break;
    i++;
    const start = i;
    // A value is one or more parts joined with #.
    while (i < body.length) {
      while (/\s/.test(body[i] ?? '')) i++;
      if (body[i] === '{') {
        const end = findClose(body, i);
        if (end === -1) return out;
        i = end + 1;
      } else if (body[i] === '"') {
        let depth = 0;
        i++;
        while (i < body.length && !(body[i] === '"' && depth === 0)) {
          if (body[i] === '\\') i++;
          else if (body[i] === '{') depth++;
          else if (body[i] === '}') depth--;
          i++;
        }
        i++;
      } else {
        const bare = /^[^\s,#}]+/.exec(body.slice(i));
        if (!bare) break;
        i += bare[0].length;
      }
      while (/\s/.test(body[i] ?? '')) i++;
      if (body[i] === '#') i++;
      else break;
    }
    out.push([name[0].toLowerCase(), body.slice(start, i).trim()]);
  }
  return out;
}

/** Joins the parts of a raw value and drops the outer braces/quotes, keeping inner LaTeX. */
function valueOf(raw: string): string {
  const parts: string[] = [];
  let i = 0;
  while (i < raw.length) {
    while (/[\s#]/.test(raw[i] ?? '')) i++;
    if (i >= raw.length) break;
    if (raw[i] === '{') {
      const end = findClose(raw, i);
      parts.push(raw.slice(i + 1, end));
      i = end + 1;
    } else if (raw[i] === '"') {
      let j = i + 1;
      let depth = 0;
      while (j < raw.length && !(raw[j] === '"' && depth === 0)) {
        if (raw[j] === '\\') j++;
        else if (raw[j] === '{') depth++;
        else if (raw[j] === '}') depth--;
        j++;
      }
      parts.push(raw.slice(i + 1, j));
      i = j + 1;
    } else {
      const bare = /^[^\s#]+/.exec(raw.slice(i))![0];
      parts.push(MONTHS[bare.toLowerCase()] ?? bare);
      i += bare.length;
    }
  }
  return parts.join('');
}

export function parseBibtex(text: string): BibEntry[] {
  const entries: BibEntry[] = [];
  const head = /@\s*([a-zA-Z]+)\s*([{(])/g;
  let match: RegExpExecArray | null;
  while ((match = head.exec(text))) {
    const type = match[1].toLowerCase();
    const open = match.index + match[0].length - 1;
    const close = findClose(text, open);
    if (close === -1) break;
    head.lastIndex = close + 1;
    if (type === 'comment' || type === 'preamble' || type === 'string') continue;
    const body = text.slice(open + 1, close);
    const comma = body.indexOf(',');
    if (comma === -1) continue;
    const rawFields = parseFields(body.slice(comma + 1));
    entries.push({
      type,
      key: body.slice(0, comma).trim(),
      rawFields,
      fields: Object.fromEntries(rawFields.map(([name, raw]) => [name, valueOf(raw)])),
    });
  }
  return entries;
}

const ACCENTS: Record<string, string> = {
  '"': '̈', "'": '́', '`': '̀', '^': '̂', '~': '̃', '=': '̄',
  '.': '̇', u: '̆', v: '̌', H: '̋', c: '̧', r: '̊', k: '̨',
};
const SPECIALS: Record<string, string> = {
  aa: 'å', AA: 'Å', ae: 'æ', AE: 'Æ', oe: 'œ', OE: 'Œ', o: 'ø', O: 'Ø', ss: 'ß', l: 'ł', L: 'Ł', i: 'ı', j: 'ȷ',
};

const MATH_SYMBOLS: Record<string, string> = {
  alpha: 'α', beta: 'β', gamma: 'γ', delta: 'δ', epsilon: 'ε', varepsilon: 'ε', zeta: 'ζ', eta: 'η',
  theta: 'θ', lambda: 'λ', mu: 'μ', pi: 'π', rho: 'ρ', sigma: 'σ', tau: 'τ', phi: 'φ', varphi: 'φ',
  chi: 'χ', psi: 'ψ', omega: 'ω', Delta: 'Δ', Gamma: 'Γ', Lambda: 'Λ', Sigma: 'Σ', Omega: 'Ω',
  ell: 'ℓ', infty: '∞', times: '×', cdot: '·', leq: '≤', le: '≤', geq: '≥', ge: '≥', pm: '±',
  to: '→', rightarrow: '→', approx: '≈', neq: '≠', nabla: '∇', partial: '∂',
};

/** Turns BibTeX/LaTeX markup into plain text: {\"a} → ä, $\ell_\infty$ → ℓ∞, -- → –, braces removed. */
export function latexToText(value: string): string {
  return value
    .replace(/\$([^$]*)\$/g, (_, math: string) =>
      math.replace(/\\([a-zA-Z]+)\s*/g, (_m, name: string) => MATH_SYMBOLS[name] ?? '').replace(/[_^{}\s]/g, ''),
    )
    .replace(/\\([\"'`^~=.]|[uvHcrk](?![a-zA-Z]))\s*\{?\s*(\\[ij]|[a-zA-Z])\s*\}?/g, (_, accent: string, letter: string) =>
      (letter === '\\i' ? 'i' : letter === '\\j' ? 'j' : letter) + ACCENTS[accent],
    )
    .replace(/\\(aa|AA|ae|AE|oe|OE|ss|o|O|l|L)(?![a-zA-Z])\s*/g, (_, name: string) => SPECIALS[name])
    .replace(/\\([&%_$#{}])/g, '$1')
    .replace(/\\[a-zA-Z]+\*?\s*/g, '')
    .replace(/[{}$]/g, '')
    .replace(/---/g, '—')
    .replace(/--/g, '–')
    .replace(/~/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .normalize('NFC');
}

/** Splits an author field on top-level "and" and returns names as "First Last". */
export function parseAuthors(field: string): string[] {
  const names: string[] = [];
  let depth = 0;
  let start = 0;
  for (let i = 0; i < field.length; i++) {
    if (field[i] === '{') depth++;
    else if (field[i] === '}') depth--;
    else if (depth === 0 && /^\sand\s/i.test(field.slice(i, i + 5))) {
      names.push(field.slice(start, i));
      start = i + 5;
      i += 4;
    }
  }
  names.push(field.slice(start));
  return names
    .map((name) => {
      const parts = splitTopLevel(name, ',').map((p) => p.trim());
      // "Last, First" or "Last, Jr., First"
      const ordered = parts.length === 1 ? parts[0] : parts.length === 2 ? `${parts[1]} ${parts[0]}` : `${parts[2]} ${parts[0]}, ${parts[1]}`;
      const text = latexToText(ordered);
      return text.toLowerCase() === 'others' ? 'et al.' : text;
    })
    .filter(Boolean);
}

function splitTopLevel(text: string, separator: string): string[] {
  const out: string[] = [];
  let depth = 0;
  let start = 0;
  for (let i = 0; i < text.length; i++) {
    if (text[i] === '{') depth++;
    else if (text[i] === '}') depth--;
    else if (depth === 0 && text[i] === separator) {
      out.push(text.slice(start, i));
      start = i + 1;
    }
  }
  out.push(text.slice(start));
  return out;
}

/** Rebuilds an entry as tidy BibTeX, leaving out the given fields. */
export function formatEntry(entry: BibEntry, omit: string[] = []): string {
  const fields = entry.rawFields.filter(([name]) => !omit.includes(name));
  const width = Math.max(0, ...fields.map(([name]) => name.length));
  const lines = fields.map(([name, raw]) => `  ${name.padEnd(width)} = ${raw}`);
  return `@${entry.type}{${entry.key},\n${lines.join(',\n')}\n}`;
}
