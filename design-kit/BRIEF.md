# Build brief: Kayla Zhang personal site

Paste this whole file into Claude Code as your first message, or tell it:
"Read `design-kit/BRIEF.md` and follow it."

---

## 0. How to work

- Read this brief and every file in `design-kit/` before writing code.
- Show a short plan first: file tree, dependencies, and the order you will build in. Wait for my OK.
- Build one page at a time. After each one, run `npm run dev`, open the page and compare it with the matching design file.
- Keep dependencies minimal. Use plain CSS with custom properties. Do not use Tailwind, UI kits or a ready-made theme.
- Do not change the look. If something in the design is unclear or seems wrong, ask me rather than guess.

## 1. What we are building

A static personal site and blog for a PhD researcher in adversarial ML and mechanistic interpretability.

- **Stack:** Astro (latest stable), static output, TypeScript.
- **Hosting:** GitHub Pages at `https://<username>.github.io`, deployed by GitHub Actions.
- **Content:** blog posts are Markdown files. Publications live in one YAML file. After setup I should never need to edit layout code to add content.

## 2. The design kit (reuse it, do not redraw)

```
design-kit/
  BRIEF.md                         this file
  design/home.dc.html              homepage design (source of truth for layout and styling)
  design/header-cat.dc.html        header with the walking cat
  design/blog-post.dc.html         blog post page with the reading-progress cat
  reference/tabby-cat.svg          the cat artwork (standing pose), final
  reference/tabby-cat-sleeping.svg the cat artwork (sleeping pose, for reduced motion)
  reference/cat-header.html        WORKING vanilla HTML/CSS/JS of the header cat
  reference/cat-reading-progress.html  WORKING vanilla HTML/CSS/JS of the progress cat
```

- The `.dc.html` files were made in a design tool. Read them for **exact** colors, fonts, sizes, spacing and structure. Ignore their tool-specific bits: `<x-dc>`, `<helmet>`, `<sc-if>`, `{{holes}}`, `support.js` and the `DCLogic` script.
- The two `reference/*.html` files are tested and behave the way I want. Port them into Astro components. Keep the SVG markup, class names, keyframes and timing unchanged.
- Open the reference files in a browser before starting, to see how they should behave.

## 3. Design tokens

Put these in `src/styles/tokens.css` as CSS custom properties.

| Token | Light | Dark |
|---|---|---|
| `--bg` | `#FAF8F3` | `#151412` |
| `--ink` (text) | `#1C1B19` | `#ECE8DF` |
| `--ink-soft` (secondary body) | `#45423C` | `#C9C4B9` |
| `--muted` (meta, captions) | `#5E5A52` | `#A39E93` |
| `--rule` (lines, borders) | `#E4DFD4` | `#2E2B27` |
| `--card` (blog cards, code) | `#F2EEE5` | `#1F1D1A` |
| `--accent` (links, progress fill) | `#1F5C5A` | `#6FB3AE` |
| `--accent-hover` | `#123B3A` | `#9AD0CB` |

- **Fonts (Google Fonts):** `Newsreader` for headings, the name and article body. `Source Sans 3` for UI, nav and meta. `IBM Plex Mono` for code and the cat's label. Self-host them with `@fontsource` packages if that's simple; otherwise use a `<link>`.
- **The cat keeps its own colors in both themes.** Don't recolor it for dark mode.
- **Dark mode:** follow the system setting by default. The moon button in the header switches themes and saves the choice in `localStorage`. Set `data-theme` on `<html>` before first paint so the page doesn't flash.

## 4. Pages and routes

| Route | Page |
|---|---|
| `/` | Home |
| `/blog/` | All posts |
| `/blog/<slug>/` | Single post |
| `/publications/` | Full publication list |
| `/cv.pdf` | My CV, a static file in `public/` (a placeholder PDF for now) |
| `/rss.xml` | RSS feed of posts |

Every page shares one header (`src/components/SiteHeader.astro`):
- The left side shows my name, which links to `/`.
- The right side has Home, Blog, Publications, CV and a dark-mode button. The current page is bold ink; the others are muted.
- Under the header runs a 1px line, and the cat lives on it:
  - Home, Blog index and Publications use **patrol mode** (section 5).
  - A single blog post uses **reading-progress mode** (section 6), and the header is sticky there.
- Content column: header max-width 960px, centered. Home and index sections are about 800px wide. Article body is 680px.

### 4.1 Home (`design/home.dc.html`)

1. **Intro, filling the first screen** (`min-height: 100svh` minus the header), with everything centered:
   - A round profile photo: 188px circle with a 6px bg-colored gap and a 1px `--rule` ring. The source is `public/profile.jpg`, and a neutral placeholder circle shows until I add it.
   - My name in Newsreader 56px/500.
   - My role in small caps style: uppercase 15px, weight 600, letter-spacing 1.5px, in accent teal: "PhD Researcher · University of Oulu".
   - A one-paragraph bio in Newsreader 21px, max-width 620px.
   - Three round icon buttons (44px, 1px `--rule` border): Google Scholar, LinkedIn and GitHub. Use the SVG paths from `home.dc.html`. The URLs come from `src/config.ts` and stay `#` until I fill them in. Each button has an `aria-label`.
   - A small "Blog and publications" link with a chevron at the bottom of the first screen, anchored to `#blog`.
2. **Blog:** a centered H2, then the latest 3 posts as cards (bg `--card`, radius 8px, padding 28px 40px, centered text). Each card shows the title in Newsreader 26px, the description in 17px `--ink-soft`, and meta in 14px `--muted` as "Mon DD, YYYY · N min · Kayla Zhang". An "All posts" link sits under the cards.
3. **Publications:** a centered H2, then the entries marked `selected: true` from the YAML file. Each entry shows the title in Newsreader 20px, then authors and *venue*, then links (PDF, Code, BibTeX, shown only when present). A "Full list" link sits underneath.
4. **Footer:** "© YEAR Kayla Zhang · Oulu, Finland", centered and muted, with a top border.

There's no News section. That's deliberate.

### 4.2 Blog index `/blog/`

Same style as the home blog section, listing all non-draft posts newest first, grouped by year with a small year heading. Keep it simple.

### 4.3 Single post (`design/blog-post.dc.html`)

- A 680px column holding the title (Newsreader 46px/500), a meta line, and a body in Newsreader 20px/1.7.
- H2 is 30px. Code blocks use `--card` bg, IBM Plex Mono 14px, radius 8px, padding 20px 24px. Figures have a caption in 15px `--muted` sans.
- **Math:** `remark-math` + `rehype-katex`, with KaTeX CSS loaded only on posts.
- **Code highlighting:** Shiki, with one light and one dark theme that match the palette. Keep it calm, no neon.
- Reading time is computed from the word count (about 220 wpm).
- At the end: a references list if the post has one, then a top border with "Back to all posts" on the left and the © line on the right.
- The header is sticky and uses the reading-progress cat.

### 4.4 Publications `/publications/`

All entries grouped by year, newest first, in the same entry style as the home page. Add a "Copy BibTeX" button per entry if `bibtex` is present (copy to clipboard, with a small "Copied" state).

## 5. Cat component, patrol mode (port `reference/cat-header.html`)

`src/components/HeaderCat.astro`, used on every page except single posts.

- The cat walks along the line under the header, from the left end to the right end, turns around and walks back, forever. It never goes past the ends of the line. Its travel distance is `line width − cat width`, recalculated with a `ResizeObserver`.
- A full round trip takes 44s, and the legs swing on a 0.46s cycle. The tail always sways.
- **On hover or keyboard focus:** it stops, tilts its head up and shows the label `tabby cat 0.03 · guacamole 0.97`.
- **On click:** it hops.
- **With `prefers-reduced-motion: reduce`:** the walker is hidden, and the sleeping cat sits at the right end with "z z".
- Sizes: 64px wide on desktop, 44px under 640px. The line area is about 60px tall so the cat stays below the nav text and never covers links.
- The cat is a `<button aria-label="Pet the cat">`.

## 6. Cat component, reading-progress mode (port `reference/cat-reading-progress.html`)

`src/components/ReadingCat.astro`, used only on single posts. The header is `position: sticky; top: 0` with a solid `--bg` background.

- The cat's position shows reading progress through the **article element**: 0 at its top, 1 when its end reaches the bottom of the viewport.
- A 2px `--accent` fill grows behind the cat along the 1px `--rule` line.
- The legs move only while the page is scrolling and stop 180ms after scrolling stops. The cat faces left when scrolling up.
- Hovering shows `NN% read`.
- The track has `role="progressbar"` with `aria-valuenow` updated.
- Scroll handling uses a passive listener and `requestAnimationFrame`, and must not jank.
- Sizes: 48px wide on desktop, 36px under 640px.
- With reduced motion, the legs and tail don't animate. The cat still moves with scroll, because the user controls that.

## 7. Content model

**`src/config.ts`**
```ts
export const site = {
  name: 'Kayla Zhang',
  role: 'PhD Researcher · University of Oulu',
  location: 'Oulu, Finland',
  bio: 'I study how vision models and vision-language models fail under adversarial pressure, and what their internals reveal about why. My work sits between adversarial machine learning and mechanistic interpretability.',
  links: { scholar: '#', linkedin: '#', github: '#' },
  photo: '/profile.jpg',
  cv: '/cv.pdf',
};
```

**Blog posts** go in `src/content/blog/<slug>.md` as an Astro content collection with a zod schema:
```yaml
---
title: Why a tabby cat can look like guacamole
description: One or two sentences for the card and meta description.
date: 2026-09-25
tags: [adversarial-examples]
draft: true
---
```
Drafts are hidden in production builds and visible in `npm run dev`.

**Publications** go in `src/data/publications.yaml`:
```yaml
- title: "[Full title]"
  authors: "[Authors, with Kayla Zhang in bold via **...**]"
  venue: IEEE Intelligent Systems
  year: 2026
  selected: true
  links: { pdf: "", code: "", bibtex: "" }
```
Seed it with these four and leave the titles and authors as placeholders for me to fill in:
1. VLM adversarial robustness evaluation, *IEEE Intelligent Systems*, 2026, selected.
2. Adaptive adversarial norm spaces, *BMVC*, 2024, selected.
3. Robust 3D pose transfer, *CVPR*, 2024, selected.
4. Stable-NAE: diffusion-based naturalistic adversarial examples, submitted to *ICIP 2026*, not selected.

**Seed post:** turn the sample article in `reference/cat-reading-progress.html` into `src/content/blog/tabby-cat-guacamole.md` with `draft: true`. Use real KaTeX math, a fenced Python code block and the three references from `design/blog-post.dc.html`. I'll review it before publishing.

## 8. SEO and polish

- Every page needs a `<title>`, meta description, canonical URL, Open Graph and Twitter tags, and a favicon. The favicon is the cat's head cropped from `tabby-cat.svg`.
- Add `@astrojs/sitemap` and `@astrojs/rss`.
- Add a 404 page in the same style. The sleeping cat plus "This page wandered off." is enough.
- Accessibility: one `h1` per page, visible focus rings in `--accent`, 44px touch targets, 4.5:1 text contrast in both themes, and alt text on the photo.
- Responsive: there's no horizontal scroll at 360px. Under 640px, hide the "Home" nav link (the name already links home) and scale the headings down.
- Lighthouse: aim for 95+ on Performance, Accessibility, Best Practices and SEO.

## 9. Deploy

- `astro.config.mjs`: set `site: 'https://<username>.github.io'`, with no `base`.
- `.github/workflows/deploy.yml` uses the official `withastro/action` plus `actions/deploy-pages` and runs on push to `main`.
- Remind me to set **Settings → Pages → Source: GitHub Actions** in the repo.
- `.gitignore` covers `node_modules`, `dist` and `.astro`.

## 10. README for future me

Write `README.md` with:
- How to run it locally (`npm install`, `npm run dev`).
- How to add a blog post: copy a template file, fill in the frontmatter, set `draft: false`, push.
- How to add a publication (one YAML entry).
- Where to change the bio, links, photo and CV.
- How to tweak the cat (speed, size) and where its code lives.

## 11. Done when

- [ ] `npm run build` passes with no warnings, and `npm run preview` looks the same as dev.
- [ ] Home matches `design/home.dc.html` at 1280px, and it holds together at 390px.
- [ ] The header cat patrols within the line, stops and shows its label on hover, hops on click and sleeps under reduced motion.
- [ ] On a post, the cat tracks reading progress, the fill follows it, it turns when scrolling up and reaches the right end exactly at the end of the article.
- [ ] Dark mode works with no flash, and the choice persists.
- [ ] Adding a Markdown post with `draft: false` makes it appear on Home, the Blog index and in RSS with no other edits.
- [ ] The GitHub Actions deploy succeeds and the site is live at `https://<username>.github.io`.
