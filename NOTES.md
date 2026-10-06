# Notes to self: how this site works

Personal site and blog of Hui Kuurila-Zhang. Built with [Astro](https://astro.build), deployed to GitHub Pages at <https://huizhg.github.io>.

## What you edit

Everything you normally change is in two folders:

```
content/
  site.ts              your name, role, bio, social links, name spellings to bold
  publications.bib     your papers, as plain BibTeX
  blog/
    _template.md       copy this to start a post
    my-post.md         one Markdown file per post (+ its images next to it)
public/
  profile.jpg          your photo
  cv.pdf               your CV
  interactive/         standalone HTML pages that posts embed (served at /interactive/)
```

You don't need to touch `src/` (the site's code) to add content.

## Add a blog post

1. In `content/blog/`, copy `_template.md` and rename the copy, e.g. `my-first-post.md`. The file name becomes the address: `/blog/my-first-post/`. (Files starting with `_` are ignored.)
2. Fill in the top part (the "frontmatter"):
   ```yaml
   ---
   title: Why a tabby cat can look like guacamole
   description: One or two sentences for the card, RSS and search results.
   date: 2026-09-25
   updated: 2026-10-02    # optional, the day you last edited the post
   tags: [adversarial-examples]
   draft: false
   references:            # optional, numbered list at the end of the post
     - Goodfellow, Shlens, Szegedy. Explaining and harnessing adversarial examples. ICLR 2015.
   ---
   ```
3. Write the post in Markdown below it.
   - Math: `$inline$` and `$$ display $$` (KaTeX).
   - Code: fenced blocks with a language, e.g. ` ```python `.
   - Images: save them in `content/blog/` next to the post and write `![alt text](./figure.png "Fig. 1. Caption.")`. The quoted text becomes the caption; leave it out for no caption. Images are resized and compressed automatically.
   - Tables: a Markdown table that is too wide for a phone scrolls sideways inside its own box.
   - References: a URL in a reference becomes a link. Cite one in the text by its number in the list, `[1]` or `[3, 4]`, and each number becomes a link to that reference.
4. `draft: true` shows the post only in `npm run dev`. Set `draft: false` and push to `main` to publish. It then appears on the home page (latest 3), the Blog page and in the RSS feed.
5. When you edit a published post, add or change `updated:`. The post then shows "Updated Oct 2, 2026" after its date. Cards and the order of posts keep using `date`.

Reading time is computed from the word count (220 words per minute).

## Interactive diagrams in a post

The MI250X post (`content/blog/mi250x-anatomy.md`) embeds two standalone pages, `public/interactive/mi250x-glance.html` and `public/interactive/mi250x-anatomy.html`. Everything in `public/` is copied to the site as it is, so they are served at `/interactive/…`.

- **Embedding:** an `<iframe>` in the post with `?embed` in its `src` hides the page's own header and footer. `mi250x-anatomy.html` also takes `?embed=explore` and `?embed=hierarchy` to show one of its two parts.
- **Height and theme:** the `<script>` near the first embed in the post grows each iframe to fit its page and passes on the site's light or dark theme, also when the toggle is clicked. The `<style>` above it makes the embeds wider than the text (up to 1120px), so the diagrams fit without sideways scrolling. To embed a page in another post, copy that `<style>`, `<script>` and one `<figure class="mi250x-embed">` block.
- **Opened on their own**, the pages follow the reader's system setting. `?theme=light` or `?theme=dark` forces one.
- **Outside requests:** the pages load fonts from Google Fonts and rough.js from cdn.jsdelivr.net. Allow both if the site ever sets a Content Security Policy.
- **Figure sources:** `content/blog/mi250x-anatomy-excalidraw/` holds the editable Excalidraw file of each figure in that post (not deployed). Open one at excalidraw.com with Open, edit, export a PNG and replace `content/blog/mi250x-anatomy-figure-N.png`.

## Add a publication

Paste the paper's BibTeX (from Google Scholar, DBLP or the publisher) into `content/publications.bib` and save. That's it.

Optional extra fields change how it shows on the site. They are removed from the BibTeX that visitors copy:

```bibtex
@inproceedings{zhang2024pose,
  title     = {Robust {3D} pose transfer},
  author    = {Doe, Jane and Zhang, Hui and Smith, John},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2024},
  selected  = {true},                     % also show it on the home page
  venue     = {CVPR},                     % short venue name to show
  pdf       = {https://arxiv.org/abs/...},  % adds a "PDF" link
  code      = {https://github.com/...}      % adds a "Code" link
}
```

- Papers are sorted by year, newest first; within a year they keep the order of the file.
- Without `venue`, the site shows the `journal` or `booktitle`.
- Your name is bold. It matches the spellings in `authorNames` in `content/site.ts` (currently "Hui Kuurila-Zhang" and "Hui Zhang").
- The citation key is the link anchor: `/publications/#zhang2024pose`.
- An entry without a `title` or `year` is skipped, with a warning in the terminal.

## Projects

The Projects page (`/projects/`) and the Projects section on the home page are placeholders that say "Coming soon." Their text is in `src/pages/projects.astro` and in the `#projects` section of `src/pages/index.astro`.

## Change the bio, links, photo and CV

- **Name, role, bio, location, social links:** `content/site.ts`.
- **Photo:** replace `public/profile.jpg` with a square image (about 400×400 px keeps it fast).
- **CV:** replace `public/cv.pdf`.
- **Favicon:** `public/favicon.svg` (the cat's head).

## View counts

Views are counted by [GoatCounter](https://www.goatcounter.com) (no cookies). Your site code goes in `goatcounter` in `content/site.ts`; an empty value turns counting off.

- Every page is counted. The dashboard at `https://<code>.goatcounter.com` shows views per page and where visitors came from.
- Each post shows its count on a line under the date and your name. No other page shows one. This needs **Settings → Allow adding visitor counts on your website** turned on in GoatCounter.
- Visits on `localhost` are not counted, so `npm run dev` doesn't inflate the numbers. The count you see there is the live site's count for that post.
- The code is the script tag in `src/layouts/BaseLayout.astro` and the script at the end of `src/layouts/PostLayout.astro`.

## Run it locally

You need Node.js 22.12 or newer (24 LTS recommended).

```sh
npm install
npm run dev        # http://localhost:4321, drafts are visible here
npm run build      # production build into dist/, drafts left out
npm run preview    # serve dist/ to check the production build
```

In Astro 7, `npm run dev` and `npm run preview` run in the background. Stop them with `npx astro dev stop` / `npx astro preview stop`.

Saving a post or `publications.bib` updates the page in `npm run dev` within a second or so. If a change doesn't show up (for example after running `npm run build` while dev was running), restart with `npx astro dev stop` and `npm run dev`.

## The cat

| | File | Used on |
|---|---|---|
| Header cat (patrol) | `src/components/HeaderCat.astro` | Home, Blog, Publications, Projects, 404 |
| Reading-progress cat | `src/components/ReadingCat.astro` | Single posts |
| Cat artwork (standing) | `src/components/CatStanding.astro` | Used by the header scenes |

The header cat does something different in each section:

| Section | Scene | Name in the code |
|---|---|---|
| Home, 404 | walks from end to end and back | `walk` |
| Blog | runs after a mouse | `chase` |
| Publications | bats a ball of yarn | `play` |
| Projects | lies down and paws at a flower | `lounge` |

- **Which scene goes where:** `activities` in `src/components/SiteHeader.astro`.
- **Speed:** `--dur` in `HeaderCat.astro` is one full round trip (`44s` walking, `22s` chasing). The other scenes' timings are in their own keyframes (`bat`, `roll`, `reach`, `nod`, …).
- **Size:** `--cat-w` (64px, 44px under 640px; the reading cat is 48px / 36px).
- **Label text:** the `.tip` span in `HeaderCat.astro`.
- Hover or keyboard focus stops the scene, the cat looks up and the label appears. Clicking makes it hop.
- The cat's colors are fixed and don't change in dark mode.
- With "reduce motion" turned on in the OS, the header cat sleeps at the right end of the line in every section. The reading cat still moves with scrolling, but its legs and tail stay still.

## Where the code lives

```
src/
  content.config.ts         tells Astro where content/ is
  lib/bibtex.ts             reads publications.bib
  lib/rehype-figure.mjs     turns images with a quoted caption into figures, makes wide tables scroll
  lib/rehype-cite.mjs       links citations like [3, 4] to the post's reference list
  lib/fresh-dev-images.mjs  in `npm run dev`, makes the browser reload an image you replaced
  styles/tokens.css         colors (light + dark) and fonts
  styles/global.css         base styles
  styles/prose.css          article body, code blocks, math, figures
  styles/code-themes.mjs    syntax highlighting colors
  layouts/                  page shells (BaseLayout, PostLayout)
  components/               header, cats, cards, publication entry, footer
  pages/                    routes: /, /blog/, /blog/<slug>/, /publications/, /projects/, /rss.xml, 404
design-kit/                 the original designs and cat prototypes (not deployed)
```

## Deploy

Every push to `main` builds and deploys with `.github/workflows/deploy.yml`. One-time setup in the GitHub repo: **Settings → Pages → Build and deployment → Source: GitHub Actions**.
