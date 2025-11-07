# GAN & qGAN — The Simple Story (Non‑technical cheat sheet)

Think of two characters:

* **Generator (G)** = a chef guessing recipes. It starts by throwing random ingredients together to cook dishes.
* **Discriminator (D)** = a food critic. It tastes dishes and says "real" (from the restaurant) or "fake" (from the chef).

You keep repeating two steps:

1. Show the critic **real dishes** (from the restaurant menu) and **fake dishes** (from the chef). Train the critic to spot the difference.
2. Now train the chef so that the critic **can’t** tell the chef’s dish from the real one.

Do this over and over. Eventually the chef learns to cook dishes that taste like the real menu.

---

## What is the data here?

* **Real data (toy/classical warm‑up):** just a list of numbers (one column). Example: 16 numbers like `[0.12, -0.31, 0.77, ...]` drawn from a bell‑curve.
* **Real data (qGAN/discretized):** we turn each number into a simple **indicator row** of length 8 (for 3 qubits). Exactly **one** position is 1, the rest are 0. Example for index 3: `[0,0,0,1,0,0,0,0]`.

Why do this indicator thing? Because quantum circuits give you **bitstrings** (like 000..111). Each bitstring matches one bin. So we make real data look the same way for the critic.

---

## What goes into G and what comes out?

* **Input to G:** a batch of random noise. Think of it as 16 “ingredient lists,” each with 100 random numbers.
* **Output of G (toy/classical):** 16 fake numbers (one per list) — should start to look like the real numbers.
* **Output of G (qGAN):** 16 fake **bins** (bitstrings → indices). We turn those indices into the same 8‑long indicator rows.

---

## What does D see?

* **Toy/classical:** D sees a batch of 16 numbers. Some are **real**, some are **fake**. It must output a score close to 1 for real, 0 for fake.
* **qGAN:** D sees a batch of 16 indicator rows of length 8. Some rows come from real bins, some from fake bins. Same goal: 1 for real, 0 for fake.

D does **not** know any formulas (like “log‑normal”). It just learns from examples: these were labeled real, these were labeled fake.

---

## The two repeating training steps (in plain words)

**Step A — Train the critic (D):**

* Give D a mixed plate: a batch of **real** items (label 1) + a batch of **fake** items from G (label 0).
* If D calls a real item “fake,” it gets a big penalty.
* If D calls a fake item “real,” it also gets a big penalty.
* After an update, D gets better at telling real from fake **for the current chef**.

**Step B — Train the chef (G):**

* Make a fresh batch of fake items.
* Ask D to score them, **but** now you pretend the target is **1** ("please say real").
* This nudges G to change its recipe so that D is more likely to call its outputs real next time.

Repeat A then B, many times.

---

## Why not train D on only real or only fake?

* Only real → D learns "everything is real" (useless).
* Only fake → D learns "everything is fake" (also useless).
* Training on **both** in the same loop teaches a meaningful boundary.

---

## About the shapes (just enough to not get lost)

* **Batch size = 16**
* **Latent noise z for G:** shape `(16, 100)` → 16 rows × 100 random numbers
* **G output (toy):** shape `(16, 1)` → 16 fake numbers
* **Real batch (toy):** shape `(16, 1)` → 16 real numbers
* **Real batch (qGAN):** shape `(16, 8)` → 16 indicator rows
* **Fake batch (qGAN):** shape `(16, 8)` → 16 indicator rows from G’s bitstrings
* **D output:** shape `(16, 1)` → 16 scores in (0,1)

---

## Qiskit LogNormal vs NumPy lognormal (plain view)

* **NumPy**: gives you raw continuous numbers (no limits).
* **Qiskit’s LogNormalDistribution**: gives you **8 fixed points** (for 3 qubits) within chosen **bounds** and a **quantum circuit** that produces those probabilities when measured. It’s a neat, quantum‑friendly, discretized version.

To compare them fairly, use the **same bounds** and the **same number of points**.

---

## The 3 lines you’ll see around every update

1. **Clear old gradients** → `optimizer.zero_grad()`
2. **Compute loss and backprop** → `loss.backward()`
3. **Apply the update** → `optimizer.step()`

That’s the whole engine.

---

# GAN & qGAN — Inputs, Batches, and Training (Quick Ref)

This note is your “open next time and remember everything” guide. It covers what the data looks like, how it’s batched, what the generator (G) and discriminator (D) see, and how the training loop updates them.

---

## 1) Tensors, Dataset, DataLoader

### Continuous real data (toy/classical GAN)

```python
import numpy as np, torch
from torch.utils.data import TensorDataset, DataLoader

real_np     = np.random.normal(loc=0.0, scale=1.0, size=(10_000, 1))  # shape (N,1)
real_tensor = torch.from_numpy(real_np).float()                        # → torch tensor
dataset     = TensorDataset(real_tensor)                               # indexable dataset
loader      = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)
```

* **Tensor:** PyTorch’s n-D array (like NumPy array + autograd + device).
* **TensorDataset:** wraps one or more tensors into an indexable dataset (`dataset[i]`).
* **DataLoader:** creates an iterator that yields **mini-batches** from the dataset.

**Why `(N, 1)` not `(N,)`?**
Most models expect `(batch_size, num_features)`. For 1-D data, `num_features=1`, so `(B,1)` fits neatly.

**Example batch (toy):**

```
real_batch (16×1) =
[[-0.12],
 [ 0.45],
 [-1.03],
 [ 0.08],
 ...]
```

---

## 2) From continuous values to bins (qGAN)

Quantum circuits output **bitstrings**, so we discretize real values into **bins** and use **one-hot** vectors.

### Discretize to `num_bins = 2^n` (e.g., 8 bins for 3 qubits)

```python
import numpy as np, torch

prices = np.random.lognormal(mean=np.log(80), sigma=0.5, size=(10_000,))
counts, edges = np.histogram(prices, bins=8)                 # edges: len=9

# Map each price → bin index in {0..7}
idx = np.digitize(prices, edges[:-1], right=False)
idx = np.clip(idx, 0, 7)                                     # safe clamp

# One-hot encode → (N, 8)
onehots = torch.nn.functional.one_hot(torch.from_numpy(idx), num_classes=8).float()
dataset = TensorDataset(onehots)
loader  = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)
```

**Example one-hot row (bin index 3):**

```
[0,0,0,1,0,0,0,0]
```

**Recover index from one-hot:**

```python
indices = onehots.argmax(dim=1)  # back to 0..7
```

---

## 3) What D and G see (by project phase)

### A) Toy/classical GAN (continuous 1-D)

* **Real to D:** `(B,1)` floats from the DataLoader.
* **Latent to G:** `z ~ N(0, I)` shape `(B, latent_dim)` (e.g., `(16,100)`).
* **Fake from G:** `G(z)` → `(B,1)` floats (same shape as real).

```python
z     = torch.randn(16, 100)   # latent
fake  = G(z)                   # (16,1)
real  = next(iter(loader))[0]  # (16,1)
```

### B) qGAN (discretized)

* **Real to D:** `(B, 2^n)` **one-hot** (e.g., `(16,8)` for 3 qubits).
* **Fake to D:** sample quantum circuit → bitstrings → indices → one-hot `(B, 2^n)`.

Optionally, you can feed **soft probability vectors** (length `2^n`) instead of one-hots, but one-hots mirror the classical GAN training better.

---

## 4) One-hot vs “binary encoding”

* **One-hot**: category indicator, exactly one `1`, rest `0`. Example for 8 bins: index 5 → `[0,0,0,0,0,1,0,0]`.
* **Binary encoding** (e.g., 5 → `101`) is **not** what we give D. We use one-hots (or probability vectors).

---

## 5) Discriminator and Generator architectures (toy)

```python
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, data_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, data_dim)    # data_dim=1 for toy (continuous),
                                               # data_dim=8 for qGAN-emulation MLP
        )
    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, data_dim, hidden_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(data_dim, hidden_dim), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1), nn.Sigmoid()  # or drop Sigmoid and use BCEWithLogitsLoss
        )
    def forward(self, x):
        return self.model(x)
```

Shapes (toy: `data_dim=1`):
`z: (B,latent_dim) → G(z): (B,1) → D(G(z)): (B,1)`

---

## 6) Labels and losses

* **Targets for D:**

  * Real batch → `1`
  * Fake batch → `0`
* **Targets for G:**
  Make `D(fake)` look **real** → use `1`.

```python
criterion = nn.BCELoss()              # or nn.BCEWithLogitsLoss with D-no-sigmoid
ones  = torch.ones(B, 1)
zeros = torch.zeros(B, 1)
```

---

## 7) Training steps (one iteration)

### D-step (train discriminator on both real and fake)

```python
opt_D.zero_grad()

# Real
p_real    = D(real)                   # (B,1)
loss_real = criterion(p_real, ones)

# Fake (detach so G doesn't get gradients now)
z         = torch.randn(B, latent_dim)
fake_det  = G(z).detach()
p_fake    = D(fake_det)
loss_fake = criterion(p_fake, zeros)

loss_D = 0.5 * (loss_real + loss_fake)
loss_D.backward()
opt_D.step()
```

Why **both** real and fake?
Only-real → D learns “predict 1 for everything.” Only-fake → D learns “predict 0 for everything.” Both → meaningful boundary.

### G-step (train generator to fool D)

```python
opt_G.zero_grad()

z      = torch.randn(B, latent_dim)
fake   = G(z)                         # no detach here
p_fake = D(fake)
loss_G = criterion(p_fake, ones)      # want D(fake)→1
loss_G.backward()
opt_G.step()
```

---

## 8) Optimizers (what `opt_D` / `opt_G` are)

```python
opt_D = torch.optim.Adam(D.parameters(), lr=1e-3)
opt_G = torch.optim.Adam(G.parameters(), lr=5e-4)
```

Update rhythm is always:
**zero_grad() → backward() → step()**.
Gradients live in `param.grad` and are accumulated until `step()`.

---

## 9) Qiskit `LogNormalDistribution` vs NumPy samples

* **NumPy:** `np.random.lognormal(mean=mu, sigma=sigma, size=...)` gives **continuous** values on `(0, ∞)`.
* **Qiskit Finance:** `LogNormalDistribution(num_qubits, mu, sigma, bounds)` gives a **discrete** pmf over `2^n` values inside `bounds`, **plus a circuit** that prepares that distribution.
  To compare fairly, align **bounds** and use the same **8 (or 2^n) points** (your histogram vs Qiskit’s `values`/`probabilities`).

---

## 10) Quick shape cheat-sheet

| Phase            | To D (real)             | To D (fake)                             | To G                  |
| ---------------- | ----------------------- | --------------------------------------- | --------------------- |
| Toy (continuous) | `(B,1)` floats          | `(B,1)` floats from `G(z)`              | `z: (B,latent_dim)`   |
| qGAN (discrete)  | `(B, 2^n)` one-hot rows | `(B, 2^n)` one-hot from quantum samples | circuit params θ / z* |

* In true qGAN the generator is a **quantum circuit** with parameters θ; you **sample shots** to get bitstrings.

---

## 11) Minimal sanity prints to keep you grounded

```python
# Toy
z    = torch.randn(16, 100)
fake = G(z)
print('z', z.shape, 'fake', fake.shape)      # (16,100) → (16,1)

# qGAN (one-hot real)
batch = next(iter(loader))[0]
print('one-hot real batch', batch.shape)     # (16, 8)

# D outputs
print('D(real)->', D(batch).shape)           # (16,1)
```
