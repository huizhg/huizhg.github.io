---
# HOW TO USE: copy this file in the same folder and rename it, e.g. my-first-post.md
# (no "_" at the start). The file name becomes the address: /blog/my-first-post/
title: Post title
description: One or two sentences for the card, RSS and search results.
date: 2026-01-31
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

Put images in this folder next to the post. The text in quotes becomes the caption:

![Describe the image for screen readers](./my-first-post-figure-1.png "Fig. 1. Caption text.")
