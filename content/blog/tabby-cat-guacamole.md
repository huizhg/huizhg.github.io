---
title: Why a tabby cat can look like guacamole
description: Change every pixel by an amount too small to see, and a confident image classifier calls a tabby cat guacamole. What adversarial examples are, the simplest way to make one, and why they matter for vision-language models.
date: 2026-09-25
tags: [adversarial-examples]
draft: true
references:
  - Szegedy et al. Intriguing properties of neural networks. ICLR 2014.
  - Goodfellow, Shlens, Szegedy. Explaining and harnessing adversarial examples. ICLR 2015.
  - Athalye, Engstrom, Ilyas, Kwok. Synthesizing robust adversarial examples. ICML 2018.
---

A photo of a tabby cat goes into an image classifier. The model is confident: tabby cat. Change every pixel by an amount too small for you to see, and the same model now says guacamole, with even higher confidence.

This is an adversarial example. It is not a bug in one model. It shows up across architectures, datasets and training recipes, and it tells us something uncomfortable about what these models actually learn.

## What is an adversarial example?

Take an input $x$ that the model classifies correctly. An adversarial example is a nearby input $x'$ that the model gets wrong. "Nearby" usually means every pixel moves by at most a small budget $\varepsilon$:

$$
\lVert x' - x \rVert_\infty \le \varepsilon
$$

With $\varepsilon = 8/255$, the change is invisible on a normal screen. For an undefended classifier, it is almost always enough.

## The simplest attack: FGSM

The fast gradient sign method takes one step in the direction that increases the loss the most, per pixel:

$$
x' = x + \varepsilon \cdot \operatorname{sign}\big(\nabla_x \mathcal{L}(\theta, x, y)\big)
$$

In PyTorch it fits in five lines:

```python
def fgsm(model, x, y, eps):
    x = x.clone().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    return (x + eps * x.grad.sign()).clamp(0, 1)
```

<figure>
  <div class="figure-placeholder">[Figure: clean image, perturbation (amplified), adversarial image]</div>
  <figcaption>Fig. 1. [Caption text.]</figcaption>
</figure>

## Why this matters for vision-language models

Vision-language models inherit their image encoders from the same family of vision models. A perturbation that steers the encoder can steer everything downstream: the caption, the answer, even whether the model follows its safety training.

That is the question my research asks from the inside. Which internal features does the perturbation move, and why does moving them change the output so much?
