---
# HOW TO USE: copy this folder (_template) inside content/blog/ and rename the copy,
# e.g. my-first-post (no "_" at the start). Keep this file's name, index.md.
# The folder name becomes the address: /blog/my-first-post/
title: Post title
description: One or two sentences for the card, RSS and search results.
date: 2026-01-31
# updated: 2026-02-14  # optional: the day you last edited the post, shown next to the date
tags: []
draft: true # true = only visible in `npm run dev`. Change to false to publish.
# references:  # optional, shown as a numbered list at the end of the post
#   - Author. Title. Venue Year.
---

First paragraph.

## A section heading

Inline math like $\varepsilon$ and display math:

$$
x' = x + \varepsilon \cdot \operatorname{sign}\big(\nabla_x \mathcal{L}(\theta, x, y)\big)
$$

```python
print("code blocks are highlighted")
```

Put images in the post's folder, next to this file. The text in quotes becomes the caption:

![Describe the image for screen readers](./figure-1.png "Fig. 1. Caption text.")
