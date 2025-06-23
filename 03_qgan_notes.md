# Day 5: qGAN High-Level Notes

## 1. Hyper-parameters

* **Generator depth (reps)**: number of circuit layers (e.g. `reps=6`). More layers = more expressive circuit but longer training time.
* **Learning rate (lr)**: size of each update step (e.g. `lr=0.01`). Higher = faster but risk overshoot; lower = more stable but slower.
* **Epochs**: number of full passes over the data (e.g. `n_epochs=50`). Fewer = under-trained; too many = wasted time or overfit.

## 2. Loss function

The losses use **binary cross-entropy**, which penalizes confident mistakes heavily.

* **Discriminator loss**
  $L_D = -\sum_j \bigl[y_j \log D(x_j) + (1-y_j) \log(1 - D(g_j))\bigr]$

  * For each real sample $x_j$ ($y_j=1$), penalizes low $D(x_j)$.
  * For each fake sample $g_j$ ($y_j=0$), penalizes high $D(g_j)$.

* **Generator loss**
  $L_G = -\sum_j \log D(g_j)$

  * Penalizes whenever the discriminator correctly identifies a fake (i.e. $D(g_j)$ near 0).

## 3. Architecture sketch

```
Real data → Discriminator D → loss
                   ↑         ↓
       Generator G (quantum circuit) → fake samples
```

1. Feed real samples to D with label=1.
2. Feed G’s generated samples to D with label=0.
3. Update D to improve its real/fake classification.
4. Update G to better fool D.
5. Repeat for N epochs.

## 4. Input / output shapes

* **Data discretization**

  * Using $n$ qubits → $2^n$ discrete bins.
  * E.g. $n=3$ → 8 bins over your numeric range (e.g. \[0…7]).

* **Generator**

  * **Input**: starts from a simple quantum state — no extra noise vector. Parameters (angles) in the circuit are learned.
  * **Output**: a probability vector of length $2^n$ (shape `(2^n,)`). Repeated measurements sample bins according to these probabilities.

* **Sampling step**

  1. **Measurement = sampling** — run the generator circuit for _N_ shots (e.g. 100). Each shot collapses to one basis index \(j_k\), giving you your fake samples directly.
  2. *(Continuous targets only)* Map each index \(j_k\) to its midpoint \(m_{j_k}\) (e.g. index 3 → 4.375) to convert discrete bins into real-valued data.


* **Discriminator**

  * **Input**: a single real-valued sample (shape `(1,)`), either from true data or G’s output.
  * **Output**: a scalar probability $D(\cdot)\in[0,1]$ indicating real vs. fake.
