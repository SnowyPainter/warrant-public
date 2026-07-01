# Warrant의 수학적 근거

본 문서는 Warrant를 empirical trick이 아니라 metric-facing weighted value term 위의 permission operator로 정식화한다. 목표는 모든 경우의 성능 향상을 보장하는 것이 아니다. 목표는 다음 명제를 수학적으로 보이는 것이다.

> Warrant는 attention relevance \(\alpha_{ij}\)를 유지한 채, metric-facing weighted value term \(\alpha_{ij}v_j\)에 item-wise permission \(g_{ij}\)를 부여한다. 이 permission은 loss gradient, signal-to-noise ratio, curvature, softmax coupling, path localization 관점에서 해석 가능한 계산 단위다.

## 0. Notation

하나의 query \(i\)에 대해 표준 attention은 key-value item \(j\)의 relevance weight \(\alpha_{ij}\)를 계산하고 value를 weighted sum한다.

\[
h_i=\sum_{j=1}^{n}\alpha_{ij}v_j.
\]

Item \(j\)가 prediction representation에 더하는 기본 계산 단위는

\[
c_{ij}=\alpha_{ij}v_j
\]

이다. 본 문서에서는 이를 weighted value term이라 부른다.

Warrant는 같은 attention weight를 유지하고, 각 weighted value term에 permission gate를 곱한다.

\[
\psi_{ij}=w^\top\mathrm{GELU}(W_q q_i+W_k k_j)+b,
\]

\[
g_{ij}=\lambda+(1-\lambda)\sigma(\psi_{ij}),
\qquad
g_{ij}\in[\lambda,1],
\]

\[
h_i^W=\sum_{j=1}^{n}\alpha_{ij}g_{ij}v_j
=\sum_{j=1}^{n}g_{ij}c_{ij}.
\]

여기서 \(\lambda>0\)는 leak factor다. Warrant는 hard mask가 아니며, 모든 valid item은 최소 \(\lambda\)만큼의 gradient/value path를 유지한다.

이후 \(i\)를 생략하고 한 query에 대해 쓴다.

---

## 1. Permission이 학습하는 것: Metric-Facing Gate Gradient

### Proposition 1

Loss를 \(\mathcal{L}(h^W)\)라고 하자. Warrant gate logit \(\psi_j\)에 대한 gradient는 다음과 같다.

\[
\frac{\partial\mathcal{L}}{\partial \psi_j}
=
(1-\lambda)\sigma'(\psi_j)\alpha_j
\left\langle
\nabla_{h^W}\mathcal{L},
v_j
\right\rangle.
\]

또는 \(c_j=\alpha_jv_j\)를 쓰면,

\[
\frac{\partial\mathcal{L}}{\partial \psi_j}
=
(1-\lambda)\sigma'(\psi_j)
\left\langle
\nabla_{h^W}\mathcal{L},
c_j
\right\rangle.
\]

### Proof

Warrant output은

\[
h^W=\sum_l \alpha_l g_l v_l.
\]

\(\psi_j\)는 \(g_j\)에만 직접 영향을 준다.

\[
\frac{\partial h^W}{\partial g_j}=\alpha_jv_j=c_j.
\]

또한

\[
\frac{\partial g_j}{\partial \psi_j}
=(1-\lambda)\sigma'(\psi_j).
\]

Chain rule에 의해

\[
\frac{\partial\mathcal{L}}{\partial \psi_j}
=
\left\langle
\nabla_{h^W}\mathcal{L},
\frac{\partial h^W}{\partial g_j}
\right\rangle
\frac{\partial g_j}{\partial \psi_j}
\]

\[
=
\left\langle
\nabla_{h^W}\mathcal{L},
\alpha_jv_j
\right\rangle
(1-\lambda)\sigma'(\psi_j).
\]

따라서 명제가 성립한다. \(\square\)

### Interpretation

\[
\left\langle \nabla_{h^W}\mathcal{L},c_j\right\rangle>0
\]

이면 \(c_j\) 방향 contribution이 loss를 증가시키는 방향이다. Gradient descent는 \(\psi_j\)를 낮추고 \(g_j\)를 낮춘다.

\[
\left\langle \nabla_{h^W}\mathcal{L},c_j\right\rangle<0
\]

이면 해당 contribution은 loss를 줄이는 방향이며, gate는 유지되거나 커지는 방향의 신호를 받는다.

따라서 Warrant는 weighted value term마다 loss-aligned permission signal을 받는다.

---

## 2. Harmful Contribution을 줄이면 Local Loss Upper Bound가 낮아진다

### Proposition 2

\(\mathcal{L}\)이 \(\beta\)-smooth라고 하자.

\[
\mathcal{L}(y)\le
\mathcal{L}(x)
+
\left\langle\nabla\mathcal{L}(x),y-x\right\rangle
+
\frac{\beta}{2}\|y-x\|^2.
\]

Base state를

\[
h=\sum_j c_j
\]

라고 하고, 하나의 item \(j\)만 Warrant로 scaling한다고 하자.

\[
h(g_j)=h+(g_j-1)c_j.
\]

그러면

\[
\mathcal{L}(h(g_j))
\le
\mathcal{L}(h)
+
(g_j-1)\left\langle\nabla\mathcal{L}(h),c_j\right\rangle
+
\frac{\beta}{2}(g_j-1)^2\|c_j\|^2.
\]

이 local upper bound를 최소화하는 permission은

\[
g_j^\star
=
\Pi_{[\lambda,1]}
\left(
1-
\frac{
\left\langle\nabla\mathcal{L}(h),c_j\right\rangle
}{
\beta\|c_j\|^2
}
\right).
\]

### Proof

\(x=h\), \(y=h(g_j)=h+(g_j-1)c_j\)를 smoothness inequality에 대입하면

\[
\mathcal{L}(h(g_j))
\le
\mathcal{L}(h)
+
\left\langle\nabla\mathcal{L}(h),(g_j-1)c_j\right\rangle
+
\frac{\beta}{2}\|(g_j-1)c_j\|^2.
\]

정리하면 첫 번째 식이 된다.

상수항을 제외한 upper bound를

\[
B(g_j)
=
(g_j-1)a_j
+
\frac{\beta}{2}(g_j-1)^2b_j
\]

라고 두자.

\[
a_j=\left\langle\nabla\mathcal{L}(h),c_j\right\rangle,
\qquad
b_j=\|c_j\|^2.
\]

미분하면

\[
\frac{dB}{dg_j}
=
a_j+\beta(g_j-1)b_j.
\]

Unconstrained optimum은

\[
g_j=1-\frac{a_j}{\beta b_j}.
\]

하지만 \(g_j\in[\lambda,1]\)이므로 projection을 적용한다.

\[
g_j^\star
=
\Pi_{[\lambda,1]}
\left(
1-\frac{a_j}{\beta b_j}
\right).
\]

\(\square\)

### Interpretation

Harmful contribution은

\[
\left\langle\nabla\mathcal{L}(h),c_j\right\rangle>0
\]

인 item이다. 이 경우

\[
g_j^\star<1
\]

이 되며, upper bound는 해당 contribution을 줄이는 방향으로 낮아진다.

Helpful contribution은

\[
\left\langle\nabla\mathcal{L}(h),c_j\right\rangle<0
\]

인 item이다. 이 경우 projection 때문에

\[
g_j^\star=1
\]

이 되어 contribution을 유지하는 해가 나온다.

즉 Warrant gate의 이상적 동작은 harmful weighted value term을 낮추고 helpful weighted value term을 보존하는 것이다.

---

## 3. SNR 개선 조건: Support Signal Retention vs Noise Standard-Deviation Retention

### Setup

Attention output의 metric-facing contribution을 support signal과 noisy contribution으로 분해한다.

\[
h_B
=
\sum_{j\in S}\alpha_j\mu_j
+
\sum_{j\in N}\alpha_j\xi_j.
\]

\(S\)는 support item set, \(N\)은 noisy 또는 non-support item set이다. \(\mu_j\)는 task-relevant signal projection이고, \(\xi_j\)는

\[
\mathbb{E}[\xi_j]=0,
\qquad
\mathrm{Var}(\xi_j)=\sigma_j^2
\]

인 noise라고 둔다.

Base SNR을 다음처럼 정의한다.

\[
\mathrm{SNR}_{B}
=
\frac{
\sum_{j\in S}\alpha_j\mu_j
}{
\sqrt{\sum_{j\in N}\alpha_j^2\sigma_j^2}
}.
\]

Warrant 적용 후 SNR은

\[
\mathrm{SNR}_{W}
=
\frac{
\sum_{j\in S}\alpha_jg_j\mu_j
}{
\sqrt{\sum_{j\in N}\alpha_j^2g_j^2\sigma_j^2}
}.
\]

### Theorem 3

다음 두 quantity를 정의하자.

Support signal retention:

\[
R_S
=
\frac{
\sum_{j\in S}\alpha_jg_j\mu_j
}{
\sum_{j\in S}\alpha_j\mu_j
}.
\]

Noise standard-deviation retention:

\[
R_N
=
\sqrt{
\frac{
\sum_{j\in N}\alpha_j^2g_j^2\sigma_j^2
}{
\sum_{j\in N}\alpha_j^2\sigma_j^2
}
}.
\]

그러면

\[
\mathrm{SNR}_W>\mathrm{SNR}_B
\quad\Longleftrightarrow\quad
R_S>R_N.
\]

### Proof

\[
\frac{\mathrm{SNR}_W}{\mathrm{SNR}_B}
=
\frac{
\sum_{j\in S}\alpha_jg_j\mu_j
}{
\sum_{j\in S}\alpha_j\mu_j
}
\cdot
\sqrt{
\frac{
\sum_{j\in N}\alpha_j^2\sigma_j^2
}{
\sum_{j\in N}\alpha_j^2g_j^2\sigma_j^2
}
}.
\]

첫 번째 항은 \(R_S\), 두 번째 항은 \(1/R_N\)이다.

\[
\frac{\mathrm{SNR}_W}{\mathrm{SNR}_B}
=
\frac{R_S}{R_N}.
\]

따라서

\[
\mathrm{SNR}_W>\mathrm{SNR}_B
\quad\Longleftrightarrow\quad
\frac{R_S}{R_N}>1
\quad\Longleftrightarrow\quad
R_S>R_N.
\]

\(\square\)

### Interpretation

Warrant가 support item을 완벽하게 보존할 필요는 없다. SNR이 개선되려면 support signal이 남는 비율이 noise standard deviation이 남는 비율보다 크면 된다.

\[
R_S>R_N.
\]

Signal은 \(g_j\)에 선형으로 남고, noise variance는 \(g_j^2\)로 줄어든다. 따라서 support gate가 높게 유지되고 noisy gate가 중간 정도로 낮아지는 mild evidence-aligned 상황에서는 SNR이 개선된다.

예를 들어

\[
R_S\approx 0.9,
\qquad
R_N\approx 0.5
\]

이면

\[
\frac{\mathrm{SNR}_W}{\mathrm{SNR}_B}
\approx
1.8.
\]

Support signal을 10% 잃어도 noise standard deviation을 50% 줄이면 SNR은 좋아진다.

이 조건은 Figure로도 표현할 수 있다. \(R_S>R_N\)인 영역에서는 Warrant SNR이 Base SNR보다 크고, \(R_S\le R_N\)인 영역에서는 false suppression 위험이 있다.

![SNR improvement region](/workspace/warrant/plots/outputs/snr_improvement_region.png)

PDF: [snr_improvement_region.pdf](/workspace/warrant/plots/outputs/snr_improvement_region.pdf)

Figure caption draft:

> SNR improvement condition and an illustrative Monte Carlo regime. Left: Warrant improves SNR when retained support signal \(R_S\) exceeds retained noise standard deviation \(R_N\). The diagonal line marks the boundary \(R_S=R_N\). Right: under an illustrative evidence-aligned gate distribution, sampled operating points concentrate mostly in the \(R_S>R_N\) region, indicating that SNR improvement is likely in this regime, though not guaranteed universally.

---

## 4. False Suppression에 대한 확률적 접근

### Assumption 1: Weighted Evidence-Aligned Permission

Support item과 noisy item의 gate가 서로 다른 조건부 분포에서 온다고 하자.

\[
g_j^S\sim G_S,
\qquad
g_j^N\sim G_N.
\]

SNR 조건에서 실제로 등장하는 것은 단순 평균이 아니라 attention weight, support signal strength, noise variance가 반영된 weighted average다. 따라서 support-side weight와 noise-side weight를 다음처럼 둔다.

\[
w_j^S
=
\frac{\alpha_j\mu_j}{\sum_{l\in S}\alpha_l\mu_l},
\qquad
\sum_{j\in S}w_j^S=1,
\]

\[
w_j^N
=
\frac{\alpha_j^2\sigma_j^2}{\sum_{l\in N}\alpha_l^2\sigma_l^2},
\qquad
\sum_{j\in N}w_j^N=1.
\]

Weighted support gate mean과 weighted noisy gate RMS를

\[
\bar g_S
=
\mathbb{E}_{w^S}[g_j]
=
\sum_{j\in S}w_j^S\mathbb{E}[g_j],
\]

\[
\bar g_N^{\mathrm{rms}}
=
\sqrt{\mathbb{E}_{w^N}[g_j^2]}
=
\sqrt{
\sum_{j\in N}w_j^N\mathbb{E}[g_j^2]
}
\]

로 정의한다. Warrant가 완벽할 필요는 없다. 다음의 약한 weighted alignment margin만 있으면 된다.

\[
\Delta
=
\bar g_S-\bar g_N^{\mathrm{rms}}
>0.
\]

즉 support item의 weighted average gate가 noisy item gate의 weighted RMS보다 크면 된다.

### Proposition 4: High-Probability SNR Improvement under Evidence-Aligned Permission

Gate가 bounded random variable이고, \(R_S\)와 \(R_N^2\)의 weighted averages가 각각의 expectation 주변으로 concentration한다고 하자. Weighted evidence-alignment margin \(\Delta>0\)가 있으면, SNR degradation event

\[
\{R_S\le R_N\}
\]

의 확률은 support/noise effective item count가 커질수록 감소한다.

### Proof Sketch

Support retention은 weighted average다.

\[
R_S=\sum_{j\in S}w_j^Sg_j,
\qquad
\sum_{j\in S}w_j^S=1.
\]

Noise variance retention은

\[
R_N^2=\sum_{j\in N}w_j^Ng_j^2,
\qquad
\sum_{j\in N}w_j^N=1.
\]

Leak-sigmoid gate 때문에

\[
g_j\in[\lambda,1],
\qquad
g_j^2\in[\lambda^2,1].
\]

따라서 \(R_S\)와 \(R_N^2\)는 bounded weighted averages다. Effective sample size를

\[
n_{\mathrm{eff}}^S=\frac{1}{\sum_{j\in S}(w_j^S)^2},
\qquad
n_{\mathrm{eff}}^N=\frac{1}{\sum_{j\in N}(w_j^N)^2}
\]

로 두면 weighted Hoeffding inequality에 의해 \(R_S\)와 \(R_N^2\)는 각각의 expectation 주변으로 concentration한다. Assumption 1에 의해 expectation level에서 positive margin이 있으므로, \(n_{\mathrm{eff}}^S,n_{\mathrm{eff}}^N\)이 커질수록

\[
\mathbb{P}(R_S\le R_N)
\]

이 작아진다. 따라서 Warrant가 SNR을 악화시키는 false-suppression event는 effective item count가 커질수록 더 낮은 확률로 발생한다. \(\square\)

### Interpretation

이 정리는 “항상 개선된다”가 아니라 다음을 말한다.

> Gate가 support item에서 평균적으로 조금 더 크고, noisy item에서 RMS 기준 조금 더 작으면, 많은 key-value item이 있는 attention setting에서 SNR 개선 확률은 높아진다.

반례도 명확하다. 만약 support gate가 모두 낮고 noisy gate가 모두 높으면 \(R_S<R_N\)이 되어 SNR은 악화된다. 따라서 이론적 claim은 universal guarantee가 아니라 evidence-aligned condition 아래의 high-probability SNR improvement다.

---

## 5. Gradient Noise Reduction

### Proposition 5

Noisy item \(j\in N\)의 value gradient noise를 \(\xi_j\)라고 하자. Base에서 value gradient noise scale은

\[
\alpha_j\xi_j
\]

이고, Warrant에서는

\[
\alpha_jg_j\xi_j
\]

이다. 따라서

\[
\mathrm{Var}[\alpha_jg_j\xi_j]
=
\alpha_j^2g_j^2\mathrm{Var}[\xi_j]
\le
\alpha_j^2\mathrm{Var}[\xi_j].
\]

### Proof

\(g_j\in[\lambda,1]\)이므로 \(g_j^2\le 1\). 상수 배율에 대한 variance 성질에 의해

\[
\mathrm{Var}[\alpha_jg_j\xi_j]
=
\alpha_j^2g_j^2\mathrm{Var}[\xi_j].
\]

따라서

\[
\mathrm{Var}[\alpha_jg_j\xi_j]
\le
\alpha_j^2\mathrm{Var}[\xi_j].
\]

\(\square\)

### Interpretation

Noisy item에 대해 \(g_j<1\)이면 해당 path의 gradient noise variance는 \(g_j^2\)만큼 감소한다. Support item의 \(g_j\)가 1에 가까우면 useful signal gradient는 대부분 유지된다. Theorem 3의 \(R_S>R_N\) 조건은 이 noise reduction이 support signal loss보다 큰 경우를 정확히 표현한다.

---

## 6. Down-Weighted Value Path의 Effective Curvature Reduction

### Proposition 6

\(\mathcal{L}\)이 \(h\)에 대해 \(\beta\)-smooth라고 하자. Item value \(v_j\) 방향의 effective Hessian norm은 Base에서

\[
\left\|
\nabla_{v_j}^2\mathcal{L}
\right\|
\le
\alpha_j^2\beta
\]

이고, Warrant에서는

\[
\left\|
\nabla_{v_j}^2\mathcal{L}
\right\|
\le
\alpha_j^2g_j^2\beta
\le
\alpha_j^2\beta.
\]

### Proof

Base에서

\[
h=\sum_l\alpha_lv_l.
\]

따라서

\[
\frac{\partial h}{\partial v_j}=\alpha_j I.
\]

Chain rule에 의해

\[
\nabla_{v_j}^2\mathcal{L}
=
\alpha_j^2\nabla_h^2\mathcal{L}.
\]

\(\|\nabla_h^2\mathcal{L}\|\le\beta\)이므로

\[
\|\nabla_{v_j}^2\mathcal{L}\|
\le
\alpha_j^2\beta.
\]

Warrant에서는

\[
\frac{\partial h^W}{\partial v_j}=\alpha_jg_jI
\]

이므로

\[
\nabla_{v_j}^2\mathcal{L}
=
\alpha_j^2g_j^2\nabla_{h^W}^2\mathcal{L}.
\]

따라서

\[
\|\nabla_{v_j}^2\mathcal{L}\|
\le
\alpha_j^2g_j^2\beta
\le
\alpha_j^2\beta.
\]

\(\square\)

### Interpretation

Warrant는 full network의 global Hessian norm이 감소한다고 보장하지 않는다. 이 명제는 gated value path를 통해 전달되는 curvature만 제한한다. Down-weighted weighted value direction에서 effective curvature upper bound가 줄어들며, noisy contribution이 큰 direction에서 \(g_j<1\)이면 해당 path의 optimization curvature가 낮아진다.

---

## 7. Warrant vs Attention-Logit Gate: Softmax Mass Redistribution이 없음

### Proposition 7

Warrant mass를

\[
m_j^W=\alpha_jg_j
\]

라고 하자. Gate logit \(\psi_l\)에 대한 derivative는 diagonal이다.

\[
\frac{\partial m_j^W}{\partial \psi_l}
=
\alpha_j(1-\lambda)\sigma'(\psi_j)\mathbf{1}[j=l].
\]

반면 logit-gated attention

\[
\tilde{\alpha}_j
=
\mathrm{softmax}_j(s_j+\log g_j)
\]

에서는

\[
\frac{\partial\tilde{\alpha}_j}{\partial \log g_l}
=
\tilde{\alpha}_j(\mathbf{1}[j=l]-\tilde{\alpha}_l).
\]

즉 attention-logit gate는 item 간 cross-coupling을 만든다.

### Proof

Warrant에서 \(\alpha_j\)는 permission logit \(\psi_l\)에 직접 의존하지 않는다.

\[
m_j^W=\alpha_jg_j.
\]

따라서 \(l\ne j\)이면 derivative는 0이고, \(l=j\)이면

\[
\frac{\partial m_j^W}{\partial \psi_j}
=
\alpha_j\frac{\partial g_j}{\partial \psi_j}
=
\alpha_j(1-\lambda)\sigma'(\psi_j).
\]

Logit-gated attention은 softmax derivative를 사용한다.

\[
\frac{\partial \tilde{\alpha}_j}{\partial z_l}
=
\tilde{\alpha}_j(\mathbf{1}[j=l]-\tilde{\alpha}_l)
\]

where \(z_l=s_l+\log g_l\). 따라서

\[
\frac{\partial\tilde{\alpha}_j}{\partial\log g_l}
=
\tilde{\alpha}_j(\mathbf{1}[j=l]-\tilde{\alpha}_l).
\]

\(\square\)

### Interpretation

Warrant는 item \(j\)의 permission을 줄여도 다른 item의 relevance mass를 강제로 올리지 않는다. Attention-logit gate는 softmax normalization 때문에 한 item의 logit을 낮추면 다른 item의 normalized attention mass가 증가한다. 따라서 attention-logit gate는 relevance distribution을 다시 정의하고, Warrant는 relevance를 유지한 채 contribution만 조절한다.

---

## 8. Post-Attention Gate는 Item-Wise Permission을 복원할 수 없다

### Proposition 8

두 decomposition이 같은 aggregate representation을 만든다고 하자.

\[
h=\sum_j\alpha_jv_j=\sum_j\alpha'_jv'_j.
\]

Post-attention gate \(F(h)\)는 두 decomposition을 구분할 수 없다. 반면 Warrant는 \(g(q,k_j)\)를 item별로 계산하므로 decomposition이 다르면 다른 permission output을 만들 수 있다.

### Proof

Post-attention gate는 aggregate \(h\)만 입력으로 받는다.

\[
\tilde{h}=F(h).
\]

두 decomposition이 같은 \(h\)를 만들면 \(F(h)\)는 동일한 output을 내야 한다. 따라서 어떤 item이 support이고 어떤 item이 noisy인지 decomposition-level 정보를 사용할 수 없다.

Warrant는 aggregation 전 item-wise term에 적용된다.

\[
h^W=\sum_j\alpha_jg(q,k_j)v_j.
\]

따라서 같은 aggregate \(h\)를 만드는 경우에도 \(k_j,v_j\)의 구성이 다르면 \(g(q,k_j)\)가 달라지고, output도 달라질 수 있다. \(\square\)

### Interpretation

이 명제는 Warrant가 일반 post-attention GLU 또는 FFN gating과 다른 이유를 보여준다. Warrant는 item index \(j\)가 사라지기 전에 weighted value term을 gate한다.

---

## 9. Path Localization Gradient: Metric에 닿는 경로에서만 Permission이 강하게 학습된다

### Proposition 9

Prediction metric이 logit 또는 score \(z\)에서 계산되고, gated value path \(h_p\)가

\[
z=f(h_p)
\]

로 metric object에 연결된다고 하자.

\[
h_p=\sum_j\alpha_jg_jv_j.
\]

그러면 gate logit gradient는

\[
\frac{\partial\mathcal{L}}{\partial\psi_j}
=
(1-\lambda)\sigma'(\psi_j)
\left\langle
J_p^\top\nabla_z\mathcal{L},
\alpha_jv_j
\right\rangle,
\]

where

\[
J_p=\frac{\partial z}{\partial h_p}.
\]

따라서 gate learning signal은 path-to-metric Jacobian \(J_p\)에 비례한다.

### Proof

Chain rule에 의해

\[
\frac{\partial\mathcal{L}}{\partial\psi_j}
=
\left\langle
\nabla_z\mathcal{L},
\frac{\partial z}{\partial h_p}
\frac{\partial h_p}{\partial g_j}
\frac{\partial g_j}{\partial\psi_j}
\right\rangle.
\]

\[
\frac{\partial h_p}{\partial g_j}=\alpha_jv_j,
\qquad
\frac{\partial g_j}{\partial\psi_j}=(1-\lambda)\sigma'(\psi_j).
\]

따라서

\[
\frac{\partial\mathcal{L}}{\partial\psi_j}
=
(1-\lambda)\sigma'(\psi_j)
\left\langle
J_p^\top\nabla_z\mathcal{L},
\alpha_jv_j
\right\rangle.
\]

\(\square\)

### Interpretation

Warrant를 metric과 약하게 연결된 attention block에 붙이면 \(J_p^\top\nabla_z\mathcal{L}\)가 작아질 수 있다. 이 경우 permission gate는 metric-facing error로부터 약한 신호만 받는다. Correct-path Warrant는 \(J_p\)가 큰 path, 즉 reported metric을 실제로 바꾸는 weighted value bottleneck을 찾는 절차다.

---

## 10. Representation Manifold 위의 Local Descent 해석

앞의 유도는 ambient Euclidean representation space에서 쓰였다. 실제 신경망의 hidden state는 고차원 공간 전체를 자유롭게 채우기보다, 데이터와 모델이 만드는 local representation manifold 근처에 놓이는 것으로 해석할 수 있다. 이때 Warrant의 수학적 의미는 ambient gradient 전체가 아니라, metric-facing state가 놓인 manifold의 tangent direction에서 정의된다.

Local \(C^2\) representation manifold를

\[
\mathcal M\subset\mathbb R^d
\]

라 하자. Prediction state \(h\in\mathcal M\)에서 tangent space를 \(T_h\mathcal M\), tangent projection을 \(P_h\)라 쓴다. Base attention output과 Warrant output을

\[
h=\sum_j c_j,
\qquad
c_j=\alpha_jv_j
\]

\[
h^W=\sum_j g_jc_j
=h+\Delta(g)
\]

로 둔다. 여기서

\[
\Delta(g)=\sum_j(g_j-1)c_j
=-\sum_j\delta_jc_j,
\qquad
\delta_j=1-g_j\in[0,1-\lambda].
\]

Manifold 위의 first-order loss change를 결정하는 것은 tangent component이다.

\[
\tilde c_j=P_hc_j,
\qquad
\xi(g)=P_h\Delta(g)
=-\sum_j\delta_j\tilde c_j.
\]

즉 Warrant의 manifold-local 효과는 weighted value term \(c_j\)를 tangent contribution \(\tilde c_j\)로 바꾼 뒤 해석할 수 있다.

### Assumption 2: Local Riemannian Smoothness

\(\mathcal L\)이 \(h\) 근방의 \(\mathcal M\) 위에서 \(\beta_{\mathcal M}\)-smooth하고, \(R_h:T_h\mathcal M\to\mathcal M\)가 local retraction이라고 하자. 그러면 충분히 작은 \(\xi\in T_h\mathcal M\)에 대해

\[
\mathcal L(R_h(\xi))
\le
\mathcal L(h)
+
\left\langle
\operatorname{grad}_{\mathcal M}\mathcal L(h),
\xi
\right\rangle
+
\frac{\beta_{\mathcal M}}{2}\|\xi\|^2.
\]

### Proposition 10: Manifold-Local Permission Rule

Single weighted value term \(c_j\)만 down-weight한다고 하자. 즉

\[
\xi_j=-\delta_j\tilde c_j,
\qquad
\delta_j\in[0,1-\lambda].
\]

다음을 정의한다.

\[
a_j=
\left\langle
\operatorname{grad}_{\mathcal M}\mathcal L(h),
\tilde c_j
\right\rangle.
\]

그러면 local upper bound는

\[
\mathcal L(R_h(-\delta_j\tilde c_j))
\le
\mathcal L(h)
-
\delta_j a_j
+
\frac{\beta_{\mathcal M}}{2}
\delta_j^2\|\tilde c_j\|^2.
\]

이 bound를 최소화하는 ideal shrinkage는

\[
\delta_j^\star
=
\Pi_{[0,1-\lambda]}
\frac{
a_j
}{
\beta_{\mathcal M}\|\tilde c_j\|^2+\epsilon
},
\qquad
g_j^\star=1-\delta_j^\star.
\]

### Proof

Assumption 2에 \(\xi=-\delta_j\tilde c_j\)를 대입한다.

\[
\mathcal L(R_h(-\delta_j\tilde c_j))
\le
\mathcal L(h)
-
\delta_j
\left\langle
\operatorname{grad}_{\mathcal M}\mathcal L(h),
\tilde c_j
\right\rangle
+
\frac{\beta_{\mathcal M}}{2}
\delta_j^2\|\tilde c_j\|^2.
\]

오른쪽은 \(\delta_j\)에 대한 1차원 convex quadratic upper bound이다. Unconstrained minimizer는

\[
\delta_j=
\frac{a_j}{\beta_{\mathcal M}\|\tilde c_j\|^2+\epsilon}
\]

이고, feasible interval \([0,1-\lambda]\)로 projection하면 위 식을 얻는다. \(\square\)

### Interpretation

\[
a_j>0
\]

이면 \(\tilde c_j\)는 manifold tangent space에서 loss-increasing direction을 향한다. 이 경우 \(\delta_j^\star>0\), 즉 \(g_j^\star<1\)이 local upper bound를 낮춘다.

\[
a_j\le0
\]

이면 해당 contribution은 helpful 또는 neutral direction이다. 이 경우 projection에 의해 \(\delta_j^\star=0\), 즉 \(g_j^\star=1\)이 된다.

따라서 Euclidean derivation의 rule은 manifold 위에서

\[
c_j\rightarrow P_hc_j,
\qquad
\nabla\mathcal L\rightarrow\operatorname{grad}_{\mathcal M}\mathcal L
\]

로 치환된 tangent-space rule로 확장된다. 이 명제는 global optimum으로의 수렴을 말하지 않는다. Warrant가 metric-facing representation manifold 위에서 loss-increasing tangent contribution을 local하게 줄이는 smooth permission interface라는 점을 말한다.

---

## 11. Tangent Gram Matrix와 Entangled Path의 한계

여러 weighted value term을 동시에 조절하면 item-wise gate는 tangent-space quadratic을 diagonal approximation으로 푸는 형태가 된다. 이때 contribution들이 tangent space에서 강하게 얽혀 있으면 Warrant가 항상 개선을 보장하지 못하는 이유도 수식으로 드러난다.

### Proposition 11: Multi-Item Manifold Bound

\[
\xi(g)=-\sum_j\delta_j\tilde c_j
\]

라 하자. 다음을 정의한다.

\[
a_j=
\left\langle
\operatorname{grad}_{\mathcal M}\mathcal L(h),
\tilde c_j
\right\rangle,
\qquad
G_{jk}=
\left\langle
\tilde c_j,
\tilde c_k
\right\rangle.
\]

그러면

\[
\mathcal L(R_h(\xi(g)))
\le
\mathcal L(h)
-
\delta^\top a
+
\frac{\beta_{\mathcal M}}{2}
\delta^\top G\delta.
\]

### Proof

Assumption 2에 \(\xi=-\sum_j\delta_j\tilde c_j\)를 대입한다.

\[
\left\langle
\operatorname{grad}_{\mathcal M}\mathcal L(h),
\xi
\right\rangle
=
-
\sum_j\delta_j
\left\langle
\operatorname{grad}_{\mathcal M}\mathcal L(h),
\tilde c_j
\right\rangle
=-\delta^\top a.
\]

또한

\[
\|\xi\|^2
=
\left\|
\sum_j\delta_j\tilde c_j
\right\|^2
=
\sum_{j,k}\delta_j\delta_k
\left\langle
\tilde c_j,\tilde c_k
\right\rangle
=\delta^\top G\delta.
\]

이를 smoothness bound에 넣으면 된다. \(\square\)

### Interpretation

\(G\)의 diagonal term은 item별 tangent norm이다. Off-diagonal term \(G_{jk}\)는 서로 다른 weighted value term이 같은 tangent direction을 공유하는 정도다. Off-diagonal mass가 작으면 item-wise Warrant gate는 좋은 diagonal shrinkage approximation이 된다.

반대로 off-diagonal mass가 크면 useful contribution과 shortcut/noisy contribution이 같은 tangent subspace에 섞인다. 이 경우 하나의 scalar permission \(g_j\)만으로 contribution을 독립적으로 분리하기 어렵고, conservative gate, false suppression, 또는 작은 성능 하락이 생길 수 있다. 이 해석은 negative row를 universal failure가 아니라 entangled path에서 local diagonal permission이 갖는 한계로 설명한다.

---

## 12. Smooth Nonconvex SGD 수준의 수렴성

Warrant는 hard mask가 아니다. 본 구현의 gate는 leak-sigmoid이다.

\[
g_j=\lambda+(1-\lambda)\sigma(\psi_j).
\]

따라서

\[
g_j\in[\lambda,1],
\qquad
\left|
\frac{\partial g_j}{\partial\psi_j}
\right|
\le
\frac{1-\lambda}{4}.
\]

또한 Warrant perturbation은 bounded이다.

\[
\|\Delta(g)\|
=
\left\|
\sum_j(g_j-1)c_j
\right\|
\le
(1-\lambda)\sum_j\|c_j\|.
\]

초기 \(g_j\approx1\)로 두면 Warrant model은 base attention network 근방에서 시작한다. 이후 학습은 smooth bounded scaling을 통해 metric-facing value path를 조절한다.

### Proposition 12: Standard First-Order Stationary Convergence

모든 학습 parameter를 \(\Theta\)라 하고, Warrant가 포함된 objective를

\[
J(\Theta)=
\mathbb E_{(x,y)}
[
\ell(F_W(x;\Theta),y)
]
\]

라 하자. 고려하는 compact parameter region에서 \(J\)가 \(L_W\)-smooth이고, stochastic gradient가 unbiased이며 variance가 \(\sigma^2\)로 bounded라고 하자.

\[
\mathbb E[g_t|\Theta_t]=\nabla J(\Theta_t),
\qquad
\mathbb E\|g_t-\nabla J(\Theta_t)\|^2\le\sigma^2.
\]

SGD update

\[
\Theta_{t+1}=\Theta_t-\eta g_t,
\qquad
\eta\le\frac1{L_W}
\]

에 대해

\[
\frac1T
\sum_{t=0}^{T-1}
\mathbb E
\|\nabla J(\Theta_t)\|^2
\le
\frac{2(J(\Theta_0)-J_\star)}{\eta T}
+
\eta L_W\sigma^2.
\]

\(\eta=O(T^{-1/2})\)이면

\[
\min_{t<T}
\mathbb E
\|\nabla J(\Theta_t)\|^2
=O(T^{-1/2}).
\]

### Proof Sketch

\(L_W\)-smoothness에 의해

\[
J(\Theta_{t+1})
\le
J(\Theta_t)
-
\eta
\left\langle
\nabla J(\Theta_t),
g_t
\right\rangle
+
\frac{L_W\eta^2}{2}\|g_t\|^2.
\]

조건부 기대값을 취하고 unbiasedness와 bounded variance를 사용하면

\[
\mathbb E[J(\Theta_{t+1})]
\le
\mathbb E[J(\Theta_t)]
-
\eta
\left(1-\frac{L_W\eta}{2}\right)
\mathbb E\|\nabla J(\Theta_t)\|^2
+
\frac{L_W\eta^2}{2}\sigma^2.
\]

\(\eta\le1/L_W\)이면

\[
\mathbb E[J(\Theta_{t+1})]
\le
\mathbb E[J(\Theta_t)]
-
\frac{\eta}{2}
\mathbb E\|\nabla J(\Theta_t)\|^2
+
\frac{L_W\eta^2}{2}\sigma^2.
\]

이를 \(t=0,\dots,T-1\)에 대해 telescoping하면 명제가 따른다. \(\square\)

### Interpretation

이 명제는 Warrant가 base보다 더 좋은 global optimum에 수렴한다고 말하지 않는다. Warrant가 smooth bounded neural-network parameterization을 유지하므로, 표준 smooth nonconvex training에서 first-order stationary point로 수렴하는 usual guarantee가 그대로 적용된다는 뜻이다.

고차원 manifold 관점에서 Warrant의 추가 의미는 더 제한적이고 더 정확하다. Warrant는 metric-facing tangent contribution에 대해 local descent-aligned shrinkage signal을 제공한다. Contribution들이 tangent Gram matrix에서 약하게 얽혀 있으면 이 signal은 깨끗하게 작동하고, 강하게 얽혀 있으면 path entanglement가 성능 혼재를 만들 수 있다.

---

## 13. 실제 실험과 수식의 연결

수식은 다음의 empirical diagnostic과 연결된다.

### BERT HotpotQA

`experiments/bert_hotpotqa_warrant/outputs/analysis/summary.csv` 기준:

\[
\mathrm{Gold/Random}\ \alpha=0.8663,
\qquad
\mathrm{Gold/Random}\ \alpha g=3.8016.
\]

Raw attention 기준으로 gold candidate가 random candidate보다 강하지 않지만, permission을 곱한 effective mass에서는 gold/random ratio가 크게 증가한다. 이는 \(R_S>R_N\) 조건의 empirical trace다.

### RAG Mass Diagnostic

RAG에서는 support attention ratio가 크게 변하지 않아도 distractor warranted mass가 감소한다. 이는 relevance \(\alpha\)를 다시 정의하지 않고 contribution permission \(g\)를 통해 effective mass를 조절하는 현상이다.

### Path Localization

Path localization 결과에서 Correct-path Warrant는 Base보다 모든 domain에서 direction-aware 성능이 높다. CTDG와 TKG에서는 Generic q-k placement보다 Correct-path placement가 크게 높다. 이는 Proposition 9의 \(J_p\) 해석과 맞다.

### Manifold-Local Simulation

`math/outputs/manifold_local_simulation.md`는 tangent-space bound

\[
B(\delta)-B(0)
=
-\delta^\top a
+
\frac{\beta_{\mathcal M}}{2}\delta^\top G\delta
\]

를 Monte Carlo로 확인한다. Low/moderate/high entanglement regime에서는 local bound가 개선되지만, tangent cross penalty가 diagonal reduction 대비 \(0.180\rightarrow0.209\rightarrow0.360\)으로 증가하면서 margin이 줄어든다. High-curvature entangled counterexample에서는 cross/diagonal ratio가 \(1.766\)으로 커져 local bound가 악화된다.

이 결과는 Warrant가 high-dimensional manifold 위에서 smooth local descent interface를 제공하지만, tangent contribution이 강하게 얽히고 local curvature가 큰 regime에서는 성능 개선을 보장하지 않는다는 해석을 뒷받침한다.

---

## 14. Reproducible Math Scripts

다음 스크립트들은 수식과 실험 결과의 정량적 연결을 확인한다.

| Script | Purpose | Output |
| --- | --- | --- |
| `math/snr_simulation.py` | evidence-aligned gate 조건에서 \(\mathbb{P}(\mathrm{SNR}_W>\mathrm{SNR}_B)\) Monte Carlo 확인 | `math/outputs/snr_simulation.csv`, `math/outputs/snr_simulation.md` |
| `math/empirical_support.py` | BERT/mass/path-localization 결과에서 ratio, error reduction, path gain을 요약 | `math/outputs/empirical_support.md`, `math/outputs/empirical_support.csv` |
| `math/permission_derivatives.py` | Warrant diagonal derivative와 attention-logit softmax coupling derivative 비교 | `math/outputs/permission_derivatives.md`, `math/outputs/permission_derivatives.csv` |
| `math/manifold_local_simulation.py` | tangent-space local bound와 tangent Gram entanglement가 Warrant 안정성에 미치는 영향 확인 | `math/outputs/manifold_local_simulation.md`, `math/outputs/manifold_local_simulation.csv` |
| `math/convergence_bound.py` | leak-sigmoid derivative bound와 smooth nonconvex stationary convergence bound 정리 | `math/outputs/convergence_bound.md`, `math/outputs/convergence_bound.csv`, `math/outputs/leak_sigmoid_bounds.csv` |

실행:

```bash
/opt/conda/bin/python math/snr_simulation.py
/opt/conda/bin/python math/empirical_support.py
/opt/conda/bin/python math/permission_derivatives.py
/opt/conda/bin/python math/manifold_local_simulation.py
/opt/conda/bin/python math/convergence_bound.py
```

---

## 15. 논문용 핵심 문장

Warrant의 이론적 역할은 universal improvement guarantee가 아니다. 다음 문장이 가장 안전하고 강하다.

> Warrant is a diagonal permission operator on metric-facing attention-weighted value terms. Its gate receives an item-wise loss-aligned gradient, and its attenuation scales both forward contribution and backward gradient exposure. On a representation manifold, the same rule applies locally after projecting weighted value terms onto the tangent space of the metric-facing state. The benefit is not unconditional: false suppression or tangent-space entanglement can reduce useful signal. However, when learned permission is mildly evidence-aligned, the retained support signal exceeds the retained noise standard deviation with high probability as effective item count grows. Unlike attention-logit gating, Warrant does not force softmax mass redistribution; unlike post-attention gates, it preserves item-wise contribution identity before aggregation.
