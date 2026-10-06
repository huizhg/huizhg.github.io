// In `npm run dev`, a resized image keeps its address when the file changes, and Astro tells the
// browser to keep it for a year. So an edited figure kept showing its old version until a hard reload.
// This makes the browser ask the dev server again each time. The built site is not affected: there
// an image's address changes with its content.

export default function freshDevImages() {
  return {
    name: 'fresh-dev-images',
    hooks: {
      'astro:server:setup': ({ server }) => {
        server.middlewares.use((req, res, next) => {
          if (req.url?.startsWith('/_image')) {
            const setHeader = res.setHeader.bind(res);
            res.setHeader = (name, value) =>
              setHeader(name, String(name).toLowerCase() === 'cache-control' ? 'no-cache' : value);
          }
          next();
        });
      },
    },
  };
}
