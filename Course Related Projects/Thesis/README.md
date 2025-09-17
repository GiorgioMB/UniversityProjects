# Graph Convolutional Networks, Random Walks, and Curvature

## Overview

This repository hosts the slide deck for a **seminar-style talk** on why **oversmoothing** and **oversquashing** arise in Graph Convolutional Networks (GCNs), and how **graph curvature** and random-walk diffusion help explain both phenomena. The talk blends spectral intuition with topological capacity arguments and closes with pragmatic advice on planning and writing a thesis.

## Context & Purpose

- **Venue:** *Topology and Machine Learning* (Università Bocconi)
- **Date:** September 2025
- **Audience:** Upper-level undergraduates and beginning graduate students in machine learning, data science, or applied probability
- **Motivation:** Provide a compact, geometry-aware account of two central GNN pathologies and a realistic assessment of when curvature-guided graph editing helps (and when it cannot).

## Slide Road‑Map

| Section                                | Slides | Highlights                                                                                              |
|----------------------------------------|:------:|----------------------------------------------------------------------------------------------------------|
| 0 Title & Context                      |   1    | Talk identity and framing                                                                                |
| 1 Roadmap                              |   2    | Two parts: (i) main technical ideas, (ii) thesis‑writing guidance                                        |
| 2 Setup                                |   3    | Affine GCN layer; random‑walk normalization; mixing limit \(A_{\mathrm{rw}}^L H^{(0)} \to \Pi H^{(0)}\) |
| 3 Positive Curvature → Oversmoothing   |   4    | Curvature ⇒ spectral gap ⇒ contraction to the stationary projector; mitigations listed                  |
| 4 Negative Curvature → Oversquashing   |   5    | Exponential neighborhood growth + narrow cuts ⇒ capacity bottlenecks; influence bound shown             |
| 5 Curvature‑Guided Editing             |   6    | Heuristic rewiring helps diagnose bottlenecks but admits no monotone “curvature‑only” objective         |
| 6 Scoping & Planning (Thesis 101)      |   7    | Scoping discipline, deliverables, and advisor workflow                                                   |
| 7 Reading, Experimenting & Final Write |   8    | Replication first, ablations that can falsify the story, reproducibility checklist, narrative spine     |
| 8 Q&A + Thesis QR                      |   9    | Pointer to the thesis via QR                                                                             |
| 9 Reference                            |  10    | Kipf–Welling (2017)                                                                                      |

*Page indices refer to the printed numbers on each slide, not the PDF page counter.*

## Key Ideas (Informal)

- **Setup.** For a graph \(G=(V,E)\) with self‑loops, define \(A_e=A+I_n\) and random‑walk normalization \(A_{\mathrm{rw}}=D_e^{-1}A_e\). An affine GCN layer is \(H^{(\ell+1)}=\sigma(A_{\mathrm{rw}} H^{(\ell)} W^{(\ell)})\). Ignoring nonlinearity and feature mixing, deep propagation behaves like multi‑hop diffusion and satisfies \(A_{\mathrm{rw}}^L H^{(0)} \to \Pi H^{(0)}\) where \(\Pi=\mathbf{1}\,\pi^{\top}\) projects onto the stationary distribution. This explains **oversmoothing** as fast mixing to a low‑rank projector.  

- **Positive curvature accelerates smoothing.** Uniformly positive (edge) curvature yields a **spectral gap** (\(\lambda_2\le1-\varepsilon\)), hence exponential contraction of components orthogonal to \(\pi\): \(\|A_{\mathrm{rw}}^L-\Pi\|_2=\lambda_2^L\). Deep stacks oversmooth earlier unless mitigated (e.g., residual/identity connections, altered normalization, decoupled propagation).

- **Negative curvature induces squashing.** When geodesic balls grow rapidly (hyperbolic‑like regions), many sources must cross **few** edges to reach a target—creating narrow cuts. For a 2‑layer GCN the average influence from the “funneling” set into the target admits an upper bound of the form \(\frac{1}{|Q_y|}\sum_{u\in Q_y} \mathrm{Inf}^{(2)}(u\to x)\le g(c(x,y),\Delta_{\max})\,\alpha^{(2)}\), where \(\alpha^{(L)}=\prod_{\ell=0}^{L-1} L_\sigma\,\|W^{(\ell)}\|_2\). This is a **topological capacity** effect, distinct from spectral mixing.

- **Curvature‑guided editing: power and limits.** Using curvature to suggest rewiring (adding edges where curvature is very negative, trimming where very positive) is a helpful **diagnostic**. But there is **no universal, strictly decreasing curvature‑only functional** that guarantees monotone improvement under non‑absorbing edits; curvature is a strong *prior*, not a complete objective.

## Using the Material

You’re welcome to cite, adapt, and extend these slides for teaching or research. If you would like the LaTeX sources or wish to report a typo, please open an issue or reach out. The QR on the penultimate slide links to the thesis that motivated this talk.

## Reference

- **Kipf, T. N. & Welling, M. (2017).** *Semi‑Supervised Classification with Graph Convolutional Networks*. arXiv:1609.02907.

## Changelog

- **v1.0 (September 2025):** Initial public release of the slides.

---

**How to cite this deck** (example):  
> Micaletto, G. (2025). *Graph Convolutional Networks, Random Walks, and Curvature: Why Oversmoothing and Oversquashing Happen*. Slides, Università Bocconi.
