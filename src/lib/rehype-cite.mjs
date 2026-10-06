// Turns numbered citations in a post into links to the post's reference list:
//   … 110 compute units [3, 4].
// becomes [<a class="cite" href="#ref-3">3</a>, <a class="cite" href="#ref-4">4</a>], where
// #ref-3 is the third entry of `references` in the frontmatter (PostLayout gives each entry its id).
// Hovering over a number shows the reference.
//
// Only numbers that have a reference are linked, so a post without `references` is left alone.
// Code, math and existing links are skipped.

const citation = /\[(\d+(?:\s*[,–-]\s*\d+)*)\]/g;

const isSkipped = (node) =>
  ['a', 'code', 'pre'].includes(node.tagName) || [node.properties?.className].flat().includes('katex');

export default function rehypeCite() {
  return (tree, file) => {
    const frontmatter = file.data.astro?.frontmatter;
    const references = Array.isArray(frontmatter?.references) ? frontmatter.references : [];
    if (references.length === 0) return;

    const link = (number) => ({
      type: 'element',
      tagName: 'a',
      properties: { className: ['cite'], href: `#ref-${number}`, title: String(references[number - 1]) },
      children: [{ type: 'text', value: number }],
    });

    const split = (text) => {
      const nodes = [];
      let last = 0;
      for (const match of text.matchAll(citation)) {
        // Numbers at even positions, the separators between them at odd ones.
        const parts = match[1].split(/(\D+)/);
        const numbers = parts.filter((_, i) => i % 2 === 0);
        if (!numbers.every((n) => Number(n) >= 1 && Number(n) <= references.length)) continue;
        nodes.push({ type: 'text', value: text.slice(last, match.index) + '[' });
        parts.forEach((part, i) => nodes.push(i % 2 ? { type: 'text', value: part } : link(part)));
        last = match.index + match[0].length - 1;
      }
      if (nodes.length === 0) return null;
      nodes.push({ type: 'text', value: text.slice(last) });
      return nodes;
    };

    const walk = (node) => {
      if (!node.children) return;
      node.children = node.children.flatMap((child) => {
        if (child.type === 'text') return split(child.value) ?? [child];
        if (child.type === 'element' && !isSkipped(child)) walk(child);
        return [child];
      });
    };
    walk(tree);
  };
}
