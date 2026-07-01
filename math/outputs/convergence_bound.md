# Warrant Smoothness and Stationary Convergence Bounds

The leak-sigmoid gate is

\[
g_j=\lambda+(1-\lambda)\sigma(\psi_j),
\]

so `g_j` is bounded and its derivative is uniformly bounded by `(1-lambda)/4`.

## Leak-Sigmoid Bounds

| gate_leak | Gate range | max abs dg/dpsi | max shrinkage |
| ---: | --- | ---: | ---: |
| 0.00 | [0.00, 1.00] | 0.2500 | 1.0000 |
| 0.01 | [0.01, 1.00] | 0.2475 | 0.9900 |
| 0.05 | [0.05, 1.00] | 0.2375 | 0.9500 |
| 0.10 | [0.10, 1.00] | 0.2250 | 0.9000 |
| 0.25 | [0.25, 1.00] | 0.1875 | 0.7500 |
| 0.50 | [0.50, 1.00] | 0.1250 | 0.5000 |

## Smooth Nonconvex SGD Bound

For an `L_W`-smooth objective with bounded stochastic-gradient variance `sigma^2`, SGD satisfies

\[
\frac1T\sum_{t=0}^{T-1}\mathbb E\|\nabla J(\Theta_t)\|^2
\le
\frac{2(J(\Theta_0)-J_*)}{\eta T}+\eta L_W\sigma^2.
\]

The numbers below instantiate the bound with objective gap `1.0`, noise variance `0.1`, and `eta=min(1/L_W,1/sqrt(T))`.

| L_W | T | eta | Average gradient norm squared bound |
| ---: | ---: | ---: | ---: |
| 1.0 | 100 | 0.10000 | 0.21000 |
| 1.0 | 300 | 0.05774 | 0.12124 |
| 1.0 | 1000 | 0.03162 | 0.06641 |
| 1.0 | 3000 | 0.01826 | 0.03834 |
| 1.0 | 10000 | 0.01000 | 0.02100 |
| 1.0 | 30000 | 0.00577 | 0.01212 |

Reading: this is not a global-optimum guarantee. It states that Warrant remains a smooth bounded neural-network parameterization, so the usual first-order stationary convergence argument still applies.
