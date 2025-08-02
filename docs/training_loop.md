# Day 26: Training Loop in Pseudocode

This document outlines step-by-step how to write and understand the training loop for our qGAN model, from basic concepts to full pseudocode. It will live at `docs/training_loop.md`.

---

## 1. Prerequisites

Before writing the loop, ensure you understand:

* **Epoch**: One full pass through the training process (real + fake batches).
* **Batch**: A subset of samples processed together (size = `batch_size`).
* **Generator (G)**: Quantum circuit that produces fake samples (probabilities over bins).
* **Discriminator (D)**: Classical neural network distinguishing real vs. fake.
* **One-hot encoding**: Converting discrete bin indices into vectors with a 1 in the chosen bin slot.
* **BCE Loss**: Binary Cross-Entropy loss for classification (real=1, fake=0).
* **Freezing parameters**: Disabling gradient updates for one network while training the other.
* **Parameter-shift gradients**: Method to obtain gradients for quantum circuit parameters by shifting them and measuring.
* **Diagnostics**: Printing losses or sample outputs every N steps to monitor training.

---

## 2. Hyperparameters

Choose values according to experiment:

* `E` (epochs): 2000
* `batch_size`: 512
* `lr_D` (Discriminator learning rate): 1e-3
* `lr_G` (Generator learning rate): 5e-3

These can be tuned later.

---

## 3. Pseudocode Outline

```python
# docs/training_loop.md

# 1. Initialize optimizers
optimizer_D = Adam(D.parameters(), lr=lr_D)
optimizer_G = Adam(G.parameters(), lr=lr_G)

# 2. Training loop
def train_loop(E, batch_size):
    for epoch in range(E):
        # 2.1 Sample a batch of real data
        real_batch = get_synthetic_distribution(batch_size)

        # 2.2 Sample a batch of fake data from G
        #    - G.sample(n_shots) returns bin indices or probabilities
        fake_probs = G.sample(batch_size)
        fake_batch = one_hot_encode(fake_probs)

        # 2.3 Update Discriminator (freeze G)
        #    - Label real as 1, fake as 0
        D.zero_grad()
        pred_real = D(real_batch)
        pred_fake = D(fake_batch.detach())
        loss_real = BCE(pred_real, torch.ones_like(pred_real))
        loss_fake = BCE(pred_fake, torch.zeros_like(pred_fake))
        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizer_D.step()

        # 2.4 Update Generator (unfreeze G)
        G.zero_grad()
        pred_fake_for_G = D(fake_batch)
        #    - Encourage D to predict 1 on fake
        loss_G = -torch.log(pred_fake_for_G).mean()
        # or: loss_G = BCE(pred_fake_for_G, torch.ones_like(pred_fake_for_G))
        loss_G.backward()  # uses parameter-shift under the hood
        optimizer_G.step()

        # 2.5 Print diagnostics every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: D_loss={loss_D.item():.4f}, G_loss={loss_G.item():.4f}")

# 3. Run training
train_loop(E=2000, batch_size=512)
```

---

## 4. Explanation of Key Steps

* **Sampling real data**: `get_synthetic_distribution(batch_size)` returns `batch_size` samples from the target (log‑normal) distribution.
* **G.sample(n\_shots)**: Runs the parameterized quantum circuit to produce raw counts or probabilities. We convert these to one-hot vectors so D can consume them as input features.
* **Detaching fake batch**: When updating D, we call `fake_batch.detach()` to prevent gradients flowing back into G.
* **BCE loss**: `BCE(pred, target)` computes `- [ target*log(pred) + (1-target)*log(1-pred) ]`.
* **Generator loss**: We want D to believe G’s outputs are real, so we maximize `log(D(G(z)))`, equivalently minimize `-log(...)`. This is implemented with parameter-shift gradients in Qiskit.
* **Parameter-shift**: Under the hood, Qiskit interpolates circuit evaluations at shifted parameter values to estimate gradients for each trainable angle.
* **Diagnostics**: Regular loss prints help detect divergence or mode collapse early.

---

## 5. Future-Hardware Note

> **Note:** When moving to real quantum hardware, replace the AerSimulator backend with `qiskit_ibm_runtime`:
>
> ```python
> from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
>
> service = QiskitRuntimeService()  # authenticate
> sampler = Sampler(session=service)
> G = QuantumGenerator(..., backend=sampler)
> ```
>
> This change allows running circuits on IBM Quantum devices with noise and real-shot execution.

---

*End of training loop guide for Day 26.*
