# Graph Convolutional Networks, Random Walks, and Curvature

## Overview

This repository hosts the slide deck for a **seminar-style talk** on why **oversmoothing** and **oversquashing** arise in Graph Convolutional Networks (GCNs), and how **graph curvature** and random-walk diffusion help explain both phenomena. The talk blends spectral intuition with topological capacity arguments and closes with pragmatic advice on planning and writing a thesis.

## Context & Purpose

- **Venue:** Università Bocconi
- **Date:** September 2025
- **Audience:** Undergraduates students currently attending the bachelor in Economics, Management and Computer Science
- **Motivation:** Provide a compact, geometry-aware account of two central GNN pathologies and a realistic assessment of when curvature-guided graph editing helps (and when it cannot).

## Key Ideas (Informal)

- **Setup.** For a graph $G=(V,E)$ with self‑loops, define $\widetilde{A}=A+I_n$ and random‑walk normalization $\widehat A_{\rm rw}=\widetilde{D}^{-1}\widetilde{A}.$ An affine GCN layer is $H^{(l+1)} = \sigma \ (\widehat A_{\rm rw} H^{(l)} W^{(l)})$. Ignoring nonlinearity and feature mixing, deep propagation behaves like multi‑hop diffusion and satisfies
$\widehat A_{\rm rw}^L H^{(0)}\to \Pi H^{(0)}$, where $\Pi=\mathbf{1}\ \pi^{\top}$ projects onto the stationary distribution. This explains **oversmoothing** as fast mixing to a low‑rank projector.  

- **Positive curvature accelerates smoothing.** Uniformly positive (edge) curvature yields a **spectral gap** ($\lambda_2\le1-\varepsilon$), hence exponential contraction of components orthogonal to $\pi$: $||A_{\mathrm{rw}}^L-\Pi||_2=\lambda_2^L$.

- **Negative curvature induces squashing.** When geodesic balls grow rapidly (hyperbolic‑like regions), many sources must cross **few** edges to reach a target—creating narrow cuts. Define $\alpha^{(L)}=\prod_{l=0}^{L-1} \mathcal L_\sigma\ ||W^{(l)}||_2$, where $\mathcal L_f$ denotes the lipschitz constant of the function $f$, then for a 2‑layer GCN the average influence from the "funneling" set into the target admits an upper bound of the form

$$\frac{1}{|Q_y|}\sum_{u\in Q_y} \mathrm{Inf}^{(2)}(u\to x)\le g(c(x,y),\Delta_{\max})\ \alpha^{(2)}.$$ 

- **Curvature‑guided editing: power and limits.** Using curvature to suggest rewiring (adding edges where curvature is very negative, trimming where very positive) is a helpful **diagnostic**. But there is **no universal, strictly decreasing curvature‑only functional** that guarantees monotone improvement under non‑absorbing edits; curvature is a strong *prior*, not a complete objective.

## Using the Material

You’re welcome to adapt and extend these slides for teaching or research. If you would like the LaTeX sources or wish to report a typo, please open an issue or reach out. The QR on the penultimate slide links to the thesis that motivated this talk and is functional until October 2025.
