# # Quantitative Monitoring for Classification Models

This document summarizes the **theory, purpose, design, and tests** for a quantitative runtime monitor that estimates how *sensitive* (or *unfair/fragile*) a model’s decisions are with respect to input changes. It complements an existing “ε-flip” monitor that reports nearby counterexamples with different decisions.

---

## 1) Purpose & intuition

Given a stream of observations $p=(x,y)$ where
- $x \in \mathbb{R}^d$ is the input (features), and
- $y \in \Delta^{k-1}$ is the predicted probability vector (softmax),

we want a **continuous** score indicating how much the model’s output can change per unit of input change, relative to previously seen points. This is a local, data-driven analogue of a **Lipschitz constant**:

$$
\boxed{\;\;q\big((x,y),(x',y')\big) \;=\; \frac{d_{\text{out}}(y,y')}{d_{\text{in}}(x,x')}\;\;}
$$

For a new point $p=(x,y)$ and a history $H$, the monitor reports

$$
Q(p;H) \;=\; \max_{(x',y')\in H} \frac{d_{\text{out}}(y,y')}{d_{\text{in}}(x,x')}.
$$

- **Large $Q$** ⇒ small input change caused a large output change (potential unfairness/instability).
- **Small $Q$** ⇒ outputs vary mildly relative to inputs (more stable).

This complements the ε-flip monitor (which answers “does there exist a nearby point with a different decision?”) with a **graded** notion (“how steep is the local slope?”).

---

## 2) Metrics & bounds

We use any input metric $d_{\text{in}}$ supported by the FRNN backend (e.g., $\ell_2,\ell_1,\ell_\infty,$ cosine) and an output metric $d_{\text{out}}$ on probability vectors. For $y,y' \in \Delta^{k-1}$, we exploit **global bounds**:

| $d_{\text{out}}$ | Global bound $b$ (for probability vectors) |
|---|---|
| $\ell_\infty$ | $b = 1$ |
| $\ell_1$ | $b = 2$ |
| Total variation $= \tfrac{1}{2}\ell_1$ | $b = 1$ |
| $\ell_2$ | $b = \sqrt{2}$ (between two one-hots) |
| cosine distance | $b = 1$ (for nonnegative vectors) |

These finite bounds are the key to **early stopping**.

> ⚠️ Divergences like KL are **unbounded**; for them we either disable early stopping or use a bounded surrogate (e.g., Jensen-Shannon).

---

## 3) Early-stoppable k-NN algorithm

Upon receiving a new input-softmax pair $(x,y)$, we must find the maximum ratio over history. Computing it against *all* past points in $H$ is $O(n)$ per observation. Instead:

1. Query for the $k$ nearest neighbors w.r.t $x$ and $d_\text{in}$ and double $k$ each round: $k=10 \to 20 \to 40 \to \dots$
2. Maintain the maximum ratio so far, $\text{max_ratio}$.
3. Let $d_k$ be the **largest input distance among the k neighbors fetched so far**
4. Use the bound $d_{\text{out}}\le b$ to conclude: **no unseen neighbor** can beat $\text{max\_ratio}$ iff

   $$
   \text{max\_ratio} \;\ge\; \frac{b}{d_k}.
   $$
   
   If this holds, we can terminate. Otherwise, return to step 1, doubling $k$.


### Correctness (proof sketch)

Assume that the algorithm terminated, that is $\text{max_ratio} \geq \frac{b}{d_k}$. Let $(x',y')$ be an unseen neighbor. By the kNN ordering, it holds that $d_\text{in}(x, z) \ge d_k$. The output distance is also bounded $d_{\text{out}}\le b$. Thus

$$
\frac{d_{\text{out}}(y,y')}{d_{\text{in}}(x,x')} \;\le\; \frac{b}{d_k} \;\le\; \text{max_ratio}.
$$


### Edge cases

- If $d_{\text{in}}(x,x') = 0$:
  - and $d_{\text{out}}(y,y') > 0$ ⇒ ratio $= +\infty$ (immediate stop).
  - and $d_{\text{out}}(y,y') = 0$ ⇒ ratio $= 0$.
- We use a tiny denominator tolerance $\tau$ to avoid float division noise: $d_{\text{in}}+\tau$.

---

## 4) API overview

### `QuantitativeMonitor`

- **Constructor**
  - `backend_factory: Callable[[], FRNNBackend]`  
    Must return a backend with **native k-NN** (`supports_knn = True`).
  - `out_metric: {"linf","l1","l2","tv","cosine"}` (default `"linf"`)  
    Chooses $d_{\text{out}}$ and its bound $b$.
  - `initial_k: int` (default 10), `max_k: Optional[int]` (cap), `tol: float` (default `1e-12`).

- **`observe(point, y, *, point_id=None, initial_k=None, max_k=None)`**
  - Compares the new pair $(x,y)$ **against history only**, returns a `QuantitativeResult`, then **inserts** the new point into the index.

- **`QuantitativeResult` fields**
  - `max_ratio: float` — the computed $Q(p;H)$ (can be `inf`).
  - `witness_id: Optional[int]` — ID achieving the max (if any), with
    - `witness_in_distance`, `witness_out_distance`.
  - `compared_count: int` — how many historical points were actually compared.
  - `k_progression: Tuple[int,...]` — $k$ values used (e.g., `10, 20, 40`).
  - `stopped_by_bound: bool` — whether early stopping triggered.
  - `point_id: int`, `note: Optional[str]`.

### Relationship to the ε-flip monitor

- **ε-flip monitor** (`clemont/monitor.py`): takes input-decision pairs (argmax of softmax) and reports *existence* of counterexamples within a fixed input radius, meaning past samples that are $\epsilon$-close but were assigned a different decision (per-decision indices are convenient there).
- **Quantitative monitor**: takes input-softmax pairs and returns a *scalar* sensitivity proxy (max ratio)

---

## 5) Testing strategy (pytests)

We wrote a comprehensive `pytest` module that does the following:

1. **Exact, in-memory backend** (`ExactBackend`) implementing true k-NN over all stored points, independent of FAISS. Ensures determinism and simplicity for tests.
2. **Hand-crafted cases** with known geometry and outputs:
   - Empty history ⇒ `max_ratio = 0`.
   - Triangular layout with softmax vectors ⇒ checks exact numeric ratio, witness distances.
   - Zero input distance with different outputs ⇒ `max_ratio = +∞`.
3. **Randomized validation vs. naïve $O(n^2)$**:
   - Generate ~**1000** samples with $x \sim \mathcal{N}$ and $y \sim \text{Dirichlet}$.
   - For each new point $i$, compare the monitor’s result against a **naïve** exact maximum over $\{0,\dots,i-1\}$.
4. Additional checks:
   - `k_progression` is monotone and starts at `initial_k`.
   - Constructor raises if the backend lacks native k-NN.

This gives **strong evidence of correctness** for both exactness (agreement with naïve) and the early-stopping logic (does not change the result).

---

## 6) Practical notes & extensions

- **Choosing $d_{\text{out}}$:**  
  $\ell_\infty$ and TV are very interpretable on probabilities. $\ell_2$ is smooth; cosine can be robust when magnitudes vary (though on probabilities magnitudes are normalized).
- **Calibration awareness:**  
  If outputs are miscalibrated, a large $Q$ might flag either genuine sensitivity or calibration noise. Pair with calibration monitoring if needed.
- **Unbounded dissimilarities:**  
  If you need KL, consider a bounded variant (JS divergence) or clip probabilities with a small floor $\epsilon$ to re-introduce a finite bound for early stopping.
- **Streaming & logging:**  
  Use `witness_id` and distances for actionable audit logs (e.g., “point 731 drove the max with $d_{\text{in}}=0.17$, $d_{\text{out}}=0.19$”).

---

## 7) TL;DR

- We track a **local Lipschitz-like** score $Q=\max d_{\text{out}}/d_{\text{in}}$ over history.
- We compute it efficiently via **doubling-k k-NN** and a **tight early-stopping bound** $b/d_k$.
- We handle **zero-distance** cases correctly and expose **witness diagnostics** for investigation.
- We validated thoroughly against **naïve $O(n^2)$** on random data and with **hand-crafted** sanity tests.
