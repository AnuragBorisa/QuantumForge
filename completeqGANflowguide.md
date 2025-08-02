# qGAN Training Overview

This comprehensive guide details every aspect of training a quantum-classical GAN (qGAN). Each section breaks down concepts, data flow, and code with rich explanations and examples so you can revisit months or years later and immediately understand the workflow.

---

## 1. Data Preparation and Data Flow

The first crucial step is transforming raw continuous samples into a form both the quantum generator (G) and classical discriminator (D) can process. Below is the detailed flow.

### 1.1 Sampling Real Data

* **Purpose:** Obtain ground-truth examples from your target distribution (e.g. log-normal).
* **How:** Use NumPy or PyTorch’s RNG:

  ```python
  import numpy as np
  real_vals = np.random.lognormal(mean=μ, sigma=σ, size=batch_size)
  # e.g. batch_size=512 → array of 512 floats
  ```
* **Key Point:** Each call draws a *random* batch—this ensures G learns the full distribution over time, not just a fixed subset.

### 1.2 Discretizing into Bins

* **Why:** Quantum circuits naturally sample discrete outcomes (bin indices). We must represent real data identically.
* **Define bin edges:** Choose `num_bins+1` edges spanning your value range:

  ```python
  min_val, max_val = real_vals.min(), real_vals.max()
  bin_edges = np.linspace(min_val, max_val, num_bins+1)
  ```
* **Map values to bins:**

  ```python
  raw_indices = np.digitize(real_vals, bin_edges, right=True) - 1
  # raw_indices[i] is integer in [0, num_bins-1]
  ```

  * `right=True` means intervals are (edge\_{i-1}, edge\_i]
  * Subtracting 1 converts 1-based digitize output to 0-based indices.
* **Example:** If `bin_edges=[0,0.5,1.0,1.5]` and `real_vals[0]=1.0`, then:

  * `digitize` gives 2 → `raw_index=1` → belongs to the second bin (0.5–1.0].

### 1.3 One-Hot Encoding

* **Problem:** D expects float vectors, not integers.
* **Solution:** Convert each integer to a binary vector:

  ```python
  import torch
  indices = torch.tensor(raw_indices, dtype=torch.long)
  real_batch = torch.nn.functional.one_hot(indices, num_classes=num_bins).float()
  # real_batch.shape == (batch_size, num_bins)
  ```
* **Interpretation:** Row `i` has a single 1 at column `raw_indices[i]`, zeros elsewhere.

### 1.4 Data Flow into D

1. **Shape check:** `real_batch` is now `(batch_size, num_bins)`.
2. **Forward pass:**

   ```python
   out_real = D(real_batch)  # shape: (batch_size, 1)
   ```
3. **Meaning:** Each output `out_real[i]` is D’s estimate (0–1) that example `i` is real.

**Summary Diagram:**

```
[continuous real_vals] --digitize--> [raw_indices] --one-hot--> [real_batch] --D--> [out_real]
```

Use this exact pipeline for fake data (with `fake_indices = G.sample(...)`).

---

## 2. Core Concepts Explained

### 2.1 Batch vs. Epoch

* **Batch (Iteration):** A subset of `batch_size` samples processed in one forward/backward pass.

  * Example: 5,120 total samples with `batch_size=512` → 10 batches per epoch.
  * **Operation:** `optimizer.zero_grad()`, compute loss on this batch, `loss.backward()`, `optimizer.step()`.
* **Epoch:** One full pass through the entire dataset (all batches).

  * After each epoch, every sample has influenced the model once.
  * Training typically runs for many epochs until convergence.

### 2.2 Generator (G)

* **Structure:** A parameterized quantum circuit with trainable angles θ₁, θ₂, …
* **Sampling:** On each call `G.sample(batch_size)`, it executes `batch_size` shots, returning bin indices or direct probability vectors.
* **Goal:** Adjust θs so that the histogram of sampled bins matches real data.

### 2.3 Discriminator (D)

* **Structure:** A classical feed-forward network (e.g., MLP) taking input size `num_bins`.
* **Output:** For each sample, outputs a scalar probability $D(x)\in(0,1)$.
* **Goal:** Maximize correct classification: $D(	ext{real})	o1$, $D(	ext{fake})	o0$.

### 2.4 One-Hot vs. Probability Vectors

* **One-Hot:** Represents a single sampled outcome (hard assignment).

  * Sharp indicator: sum of entries = 1, exactly one 1.
* **Probability Vector:** Soft distribution over bins (float values sum to 1).

  * Captures G’s full belief; can be fed directly to D.
* Both real and fake must use the *same* format in any given run.

### 2.5 Binary Cross-Entropy (BCE) Loss

* **Definition:** For a predicted score $p$ and label $t\in\{0,1\}$:

  $$
    \mathrm{BCE}(p,t) = -igl[t\log(p) + (1-t)\log(1-p)igr].
  $$
* **Applied to D:**

  * `loss_real = BCE(D(real_batch), 1)`
  * `loss_fake = BCE(D(fake_batch), 0)`
  * `loss_D = loss_real + loss_fake`.
* **Applied to G:** encourage D to predict real on fakes:

  * `loss_G = BCE(D(fake_batch), 1)` or equivalently (-\log(D(fake))\`.

### 2.6 Freezing & Detach (Detailed)

In adversarial training, each backward pass computes gradients for all connected parameters. Since both networks’ losses can flow through each other’s graphs, we explicitly control gradient flow so that:

* **D only learns from the discriminator loss**
* **G only learns from the generator loss**

#### 1. Detaching fake data in the D-step

When updating D, we must cut the link back to G so that calling `backward()` affects only D’s parameters:

```python
opt_D.zero_grad()                    # Clear D’s old gradients
fake_samples = G.sample(batch_size)  # Tensor requires_grad=True by default
fake_detached = fake_samples.detach()# New tensor: same data, requires_grad=False

out_real = D(real_batch)             # Scores for real data
out_fake = D(fake_detached)          # Scores for fake data, no G.grad

loss_D = BCE(out_real, 1) + BCE(out_fake, 0)
loss_D.backward()                    # Gradients flow into D only
opt_D.step()                         # Update D’s weights
```

* **`detach()`** severs the autograd connection, making `fake_detached` a leaf node with no gradient record.
* Gradients computed on `out_fake` stop at `fake_detached` and never reach G.

#### 2. Freezing D’s weights in the G-step

When training G, we want gradients to flow through D’s forward pass *into* G, but we don’t want D’s weights to update. Two common ways:

**A. Temporarily disable D’s parameter gradients**

```python
# Before computing G’s loss:
for p in D.parameters():
    p.requires_grad = False          # Freeze D’s weights

opt_G.zero_grad()                   # Clear G’s old gradients
fake_batch = G.sample(batch_size)   # Generate new samples
out_fake = D(fake_batch)            # Forward through D

loss_G = BCE(out_fake, 1)           # Encourage D(fake)=1
loss_G.backward()                   # Gradients flow into G only
opt_G.step()                        # Update G’s parameters

# Re-enable D’s gradients:
for p in D.parameters():
    p.requires_grad = True
```

**B. Use a no-grad context for D’s parameters**

```python
opt_G.zero_grad()
with torch.no_grad():               # Disable gradient tracking globally
    for p in D.parameters():
        p.requires_grad = False
out_fake = D(fake_batch)

loss_G = BCE(out_fake, 1)
loss_G.backward()                   # G.grad populated, D frozen
opt_G.step()
# Reset requires_grad for D as needed
```

**Why explicit control matters:**

1. **Isolation** prevents a network from accidentally updating on the other’s loss.
2. **Stability** maintains the intended adversarial dynamic: D improves on real vs. fake, G improves at fooling D.
3. **Clarity** ensures that when you read the code—even years later—you immediately see which network is learning at each step.

### 2.7 Parameter-Shift Rule

* **Challenge:** Quantum circuits aren’t differentiable in the usual way.
* **Technique:** For each parameter θ: evaluate circuit at θ±π/2.
* **Gradient:** $\partial f/\partial	heta = 	frac{1}{2}[f(	heta+	frac{\pi}{2})-f(	heta-	frac{\pi}{2})]$.
* Qiskit automates this when you call `.backward()` on a hybrid quantum-classical loss.

---

## 3. Training Loop & Data Flow

Below is the fully annotated loop, showing data and gradient flow:

```python
# --- Setup ---
opt_D = Adam(D.parameters(), lr=lr_D)   # Discriminator optimizer
dopt_G = Adam(G.parameters(), lr=lr_G)  # Generator optimizer

for epoch in range(num_epochs):
    for real_vals in data_stream(batch_size):      # 1. Sample real data
        # 2. Prepare real batch
        raw_idx     = np.digitize(real_vals, bin_edges, right=True) - 1
        real_batch  = one_hot(raw_idx)             # shape: [B, N]

        # 3. Prepare fake batch
        fake_idx    = G.sample(batch_size)         # quantum sampling
        fake_batch  = one_hot(fake_idx)            # shape: [B, N]

        # --- Discriminator update ---
        opt_D.zero_grad()                          # Clear D gradients
        out_r = D(real_batch)                      # D’s real scores [B,1]
        out_f = D(fake_batch.detach())             # detach prevents G.grad
        loss_D = BCE(out_r, 1) + BCE(out_f, 0)
        loss_D.backward()                          # grads only for D’s params
        opt_D.step()                               # update D

        # --- Generator update ---
        opt_G.zero_grad()                          # Clear G gradients
        pred    = D(fake_batch)                    # forward fake through D
        loss_G  = BCE(pred, 1)                     # want D to call fakes real
        loss_G.backward()                          # grads only for G’s params
        opt_G.step()                               # update G

    # End of epoch diagnostics
    print(f"Epoch {epoch}: D_loss={loss_D:.4f}, G_loss={loss_G:.4f}")
```

**Data Flow Summary:**

* Real data: continuous → discretize → one-hot → D → loss\_D → backward → D.update
* Fake data: G.sample → one-hot → D → loss\_G → backward → G.update

---

## 4. Why Mini-Batches?

1. **Memory constraints**

   * Full-batch (all samples) may not fit on GPU/CPU memory.
   * Mini-batches of size 32–512 are hardware-friendly.

2. **Faster convergence**

   * More frequent weight updates speed up learning.
   * Example: 10 mini-batches/epoch → 10 updates vs. 1 update.

3. **Gradient noise benefits**

   * Stochastic gradients help escape shallow minima.
   * Acts as an implicit regularizer.

4. **Parallelism**

   * GPUs excel at batching vector operations.
   * Fixed-size tensors maximize throughput.

5. **Streaming & scalability**

   * Supports datasets too large to load fully.
   * Allows online learning via continual data streams.

---

## 5. Classical GAN vs. Histogram qGAN

| Aspect              | Classical GAN                                  | Histogram qGAN                               |
| ------------------- | ---------------------------------------------- | -------------------------------------------- |
| Data representation | Continuous vectors in ℝᵈ                       | Discrete bins → one-hot or probability ports |
| Generator output    | Float tensors matching real data dims          | Bin indices or soft probability vectors      |
| Discriminator input | Raw floats                                     | One-hot/prob vectors over bins               |
| Loss objective      | Minimize divergence between continuous distros | Match histograms via adversarial training    |
| Use case            | Any continuous data (images, audio)            | Quantum sampling for statistical modeling    |

**Key takeaway:** Both follow the adversarial loop—D vs. G—but differ in how data is encoded and fed into the networks.

---

*This document is your future ten-year refresher: dive into any section to recall exactly how and why each step works.*
