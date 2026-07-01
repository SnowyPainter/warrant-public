# Manifold-Local Simulation

This diagnostic simulates Warrant as a diagonal shrinkage rule on tangent-space weighted value terms.
The local Riemannian smoothness bound is written as

\[
B(\delta)-B(0)=-\delta^\top a+\frac{\beta_{\mathcal M}}{2}\delta^\top G\delta,
\]

where `a` is the tangent loss-gradient alignment and `G` is the tangent Gram matrix of weighted value terms.
The item-wise Warrant rule uses only the diagonal of `G`; large off-diagonal mass means the path is entangled.

| Regime | P(bound improves) | Bound reduction | Diagonal reduction | Cross penalty | Cross/Diagonal | Harmful shrinkage | Helpful shrinkage | False suppression |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| low_entanglement | 1.000 | 0.3009 | 0.3678 | 0.0670 | 0.180 | 0.950 | 0.000 | 0.000 |
| moderate_entanglement | 1.000 | 0.2897 | 0.3673 | 0.0777 | 0.209 | 0.950 | 0.000 | 0.000 |
| high_entanglement | 1.000 | 0.2321 | 0.3643 | 0.1322 | 0.360 | 0.950 | 0.000 | 0.000 |
| high_curvature_entangled_counterexample | 0.000 | -0.2607 | 0.3353 | 0.5960 | 1.766 | 0.950 | 0.000 | 0.000 |

Reading: the manifold-local argument supports local stability, not global superiority.
As tangent Gram off-diagonal mass increases, the diagonal item-wise gate becomes a rougher approximation and the cross penalty grows.
The counterexample row shows that high curvature plus strong tangent entanglement can overturn the diagonal shrinkage benefit.
This is the mathematical failure mode behind entangled negative rows: useful and harmful directions can share the same tangent subspace.
