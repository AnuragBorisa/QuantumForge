# Log‑Normal, Bins, Stocks & Qiskit — Simple Notes

These are simple notes in plain language so you can quickly recall the ideas later.

---

## 1) Daily price vs daily return (what data we use)

* **Price** = the stock’s closing value each day (₹).
* **Daily return** = how much today changed **relative to yesterday**.

  * **Simple return:** `(P_today − P_yesterday) / P_yesterday` (e.g., +0.012 = +1.2%).
  * **Log return:** `ln(P_today) − ln(P_yesterday)` (nice for math; adds over days).
* We usually study the **distribution of daily returns** (not prices) because prices drift, but returns are more “stable” to summarize.

**Mini example** (prices → returns):

```
Prices: 100 → 103 → 101
Simple returns: +3.0%,  −1.94%
Log returns:    +2.956%, −1.933%
```

---

## 2) What is a bin? What is a bin center?

* A **bin** is just a bucket that covers a small **range** of returns.

  * Example bins across −5% to +5% with width 1%:

    * [−5%, −4%), [−4%, −3%), …, [0%, +1%), …, [+4%, +5%).
* The **bin center** is the **midpoint** of that range (used to draw the bar at the right x‑position).

  * Example: bin [0%, +1%) has center +0.5%.

**Meaning**: “Count how many days landed in each range.” That bar chart (histogram) is your **distribution of returns**.

---

## 3) Count → Probability

* **Count in a bin** = number of days that fell in that range.
* **Probability of a bin** = `count_in_bin / total_days`.
* All bin probabilities add up to ~1 (100%).

**Example** (100 days):

```
Bin [0%, +1%]  → 38 days  → probability 0.38 (38%)
Bin [−1%, 0%]  → 22 days  → probability 0.22 (22%)
```

---

## 4) What is “log‑normal” (plain idea)

* Take a positive number `X`. If `ln(X)` is **Normal (bell‑shaped)**, then `X` is **Log‑Normal**.
* Log‑normal values are **positive** and **right‑skewed** (long right tail).
* In finance, a **future price at a fixed horizon** under simple models can look log‑normal.

**Key parameters (of the log)**: `μ` (mean of `ln X`) and `σ` (std of `ln X`).

---

## 5) Qiskit LogNormal — what it builds

* You choose a **range** `[low, high]` and **`num_qubits = n`**.
* This creates **`2^n` equal‑width bins** across `[low, high]`.
* Qiskit computes a **probability** `p_i` for each bin `i` from the log‑normal (using your `μ, σ`).
* It prepares a **quantum state** whose measurement returns bin `i` with probability `p_i`:

  * State looks like `|ψ⟩ = Σ sqrt(p_i) |i⟩`.
* Measuring many times reproduces the **same bin frequencies** as your discretized log‑normal.

**Notes**:

* It’s **discretized** (bins), not a smooth curve.
* It’s **clipped** to `[low, high]` (choose wide enough).
* More qubits ⇒ more bins ⇒ finer shape.

---

## 6) Why Qiskit LogNormal vs NumPy log‑normal?

* **NumPy log‑normal**: great for classical **sampling and averaging** today.
* **Qiskit LogNormal**: puts the distribution **inside qubits** so you can use **Quantum Amplitude Estimation (QAE)** to estimate averages (like an option’s expected payoff) with **far fewer queries** (theoretical **quadratic speedup** in accuracy needs) — on good quantum hardware.

**Simple takeaway**:

* If you just need a number now → **NumPy**.
* If you want to **compose quantum finance blocks** and target theoretical speedups for expectations → **Qiskit LogNormal**.

---

## 7) GAN vs Qiskit LogNormal (different jobs)

* **Qiskit LogNormal**: a **prebuilt shape** (you supply `μ, σ`, bins, range). No learning.
* **(q)GAN**: a **learner** that tries to **match the histogram from real data**, even if it’s **not** log‑normal.
* Use LogNormal for **assumed/log‑normal demos**; use GAN/qGAN to **copy the real distribution** from data.

---

## 8) Option pricing — kid‑simple example

* Today price: **₹100**. Call strike: **₹102**. Payoff tomorrow: `max(price − 102, 0)`.
* We need the **average payoff** across all possible tomorrows.

**NumPy way**:

1. Sample many log‑normal prices for tomorrow.
2. Compute payoff each time; **average** them.

**Qiskit way** (idea):

1. Use **LogNormalDistribution** to put the possible prices (bins + probabilities) **into qubits**.
2. Add a small circuit that encodes the **payoff** per bin.
3. Use **Amplitude Estimation** to read the **average payoff** with fewer queries (in theory).

---

## 9) Quick recipes

* **Build returns** from prices: `r_t = (P_t − P_{t−1}) / P_{t−1}` or `ℓ_t = ln(P_t) − ln(P_{t−1})`.
* **Make bins**: choose a range (e.g., −5%…+5%), choose bin count (e.g., 20), count days per bin.
* **Probabilities**: `prob_i = count_i / total_days`.
* **Qiskit bounds**: pick `[low, high]` by taking `(μ ± 3σ)` in **log‑space**, then exponentiate: `low = exp(μ − 3σ)`, `high = exp(μ + 3σ)`.

---

## 10) Common pitfalls to remember

* **Don’t bin raw prices** to get a stationary distribution; bin **returns**.
* **Point pdf values** aren’t “chances”; chances live over **ranges (bins)**.
* **Unconditional GAN** gives **independent** one‑day samples (no volatility clustering). For realistic **paths**, use a **sequence model** (TimeGAN/diffusion) or combine returns with a volatility model (e.g., GARCH).
* **Qiskit LogNormal** is for building a **quantum state** (useful for quantum algorithms), not for faster classical sampling.

---

### One‑line memory hook

**Bins** = ranges of returns; **probability** = how often days fell in each range; **Qiskit LogNormal** = puts those range‑probabilities into qubits so quantum tools can estimate averages (like option prices) more efficiently **in theory**.
