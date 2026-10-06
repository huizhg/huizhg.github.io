// Turns an image that sits alone in a paragraph and has a title into a captioned figure:
//   ![A tabby cat](./cat.png "Fig. 1. A tabby cat.")
// becomes <figure><img …><figcaption>Fig. 1. A tabby cat.</figcaption></figure>.
//
// Also wraps a Markdown table in <div class="table-scroll">, so a table that is too wide for a
// phone scrolls sideways inside its own box instead of widening the page.

const isBlank = (node) => node.type === 'text' && !node.value.trim();

export default function rehypeFigure() {
  const walk = (node) => {
    if (!node.children) return;
    node.children = node.children.map((child) => {
      if (child.type === 'element' && child.tagName === 'table' && node.type === 'root') {
        return { type: 'element', tagName: 'div', properties: { className: ['table-scroll'] }, children: [child] };
      }
      if (child.type === 'element' && child.tagName === 'p') {
        const content = child.children.filter((c) => !isBlank(c));
        const img = content[0];
        if (content.length === 1 && img.type === 'element' && img.tagName === 'img' && img.properties?.title) {
          const caption = String(img.properties.title);
          delete img.properties.title;
          return {
            type: 'element',
            tagName: 'figure',
            properties: {},
            children: [img, { type: 'element', tagName: 'figcaption', properties: {}, children: [{ type: 'text', value: caption }] }],
          };
        }
      }
      walk(child);
      return child;
    });
  };
  return walk;
}
