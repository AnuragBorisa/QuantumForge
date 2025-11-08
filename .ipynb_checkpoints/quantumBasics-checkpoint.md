# Quantum Superposition, Measurement & Rotations — Simple Notes

These notes collect the ideas we just discussed, in **simple language** with small formulas you can glance at later. Use this as a quick refresher.

---

## 1) Qubit, bases, and inner product

* A **qubit** lives on a 2D complex vector space. The usual (Z) basis is **|0⟩, |1⟩**.
* A general state (pure) is **|ψ⟩ = α|0⟩ + β|1⟩** with |α|² + |β|² = 1.
* **Inner product** gives overlap: ⟨e|ψ⟩ is “how much |ψ⟩ points along |e⟩”.
* **Born rule (probabilities)**: measuring along a basis vector |e⟩ gives **p(e) = |⟨e|ψ⟩|²**.

**X and Y bases** (you’ll use these a lot):

* **X-basis**: |+⟩ = (|0⟩+|1⟩)/√2, |−⟩ = (|0⟩−|1⟩)/√2.
* **Y-basis**: |+i⟩ = (|0⟩+i|1⟩)/√2, |−i⟩ = (|0⟩−i|1⟩)/√2.

---

## 2) What Z, X, Y “ask” in plain words

* **Z** asks: “Which one is it, |0⟩ or |1⟩?” (populations). Phase is invisible here.
* **X** asks: “Do |0⟩ and |1⟩ **add** (bright) or **subtract** (dark) with **no extra phase**?”
* **Y** asks: “Is |1⟩ **ahead/behind by 90°** (±i) relative to |0⟩?” (lead/lag)

**Hardware trick:** devices usually measure Z. To measure X, do **H then Z**. To measure Y, do **S† then H, then Z**.

---

## 3) Single‑qubit rotations (Rx, Ry, Rz)

Think: **rotate the Bloch arrow around an axis**.

* **RZ(φ)**: changes only the **relative phase** between |0⟩ and |1⟩. Z-probabilities don’t change; weight moves between X and Y.
* **RY(θ)**: changes **how much 0 vs 1** with **no ±90° phase**. X sees the change; Y stays neutral (starting from real states).
* **RX(θ)**: also changes 0 vs 1, but **adds ±90° relative phase** so **Y** reacts (X can be neutral).

Special cases:

* **Pauli X, Y, Z** are π rotations (up to a global phase):

  * X ≃ RX(π), Y ≃ RY(π), Z ≃ RZ(π).
* **RY(π/2)** makes **|+⟩** from |0⟩; **RX(π/2)** makes **|−i⟩** from |0⟩.

**Pauli on basis** (good to memorize):

* X|0⟩=|1⟩, X|1⟩=|0⟩
* Y|0⟩= i|1⟩, Y|1⟩= −i|0⟩
* Z|0⟩=|0⟩, Z|1⟩= −|1⟩

---

## 4) Two‑qubit interactions (RZZ, RXX, RYY) — plain feel

All are **RAA(θ) = exp(−i(θ/2) A⊗A)** for A ∈ {X, Y, Z}.

* **RZZ(θ)**: adds **phases** depending on whether the bits are **same (00/11)** or **different (01/10)**. It does **not** move population; it changes interference → can **entangle** superpositions.
* **RXX(θ)**: **mixes pairs** |00⟩↔|11⟩ and |01⟩↔|10⟩ (like a beam‑splitter), with a quarter‑turn phase on the moved part → entanglement.
* **RYY(θ)**: same mixing idea as RXX with a different internal phase choice (think: “RZZ in the Y‑basis”).

Handy identities:

* **RZZ(θ) = CNOT · (I⊗RZ(θ)) · CNOT**.
* RXX and RYY are RZZ conjugated by single‑qubit basis changes (H, S gates).

---

## 5) CRZ (controlled‑RZ) — why it’s a ZZ‑style coupling

* **CRZ(λ)**: apply **RZ(λ)** on the **target only if the control is |1⟩**. It’s diagonal in the computational basis and changes phases conditionally.
* With H/S basis changes and CNOTs, CRZ is a building block for **RZZ**; both are “Z‑dependent phase” interactions used in QAOA cost layers.

Minimal pattern you’ll see in code:

```text
CRZ(λ) = CX • RZ(λ on target) • CX
```

---

## 6) Measurement — projectors, probabilities, collapse

**Projector** = “keep‑only‑this‑part” operator.

* Z‑measurement uses P0 = |0⟩⟨0|, P1 = |1⟩⟨1|.
* **Probabilities (Born rule)**: p(0)=⟨ψ|P0|ψ⟩=|α|², p(1)=|β|².
* **Collapse**: if you see 0 → state becomes |0⟩; if 1 → |1⟩.
* **Expectation**: ⟨Z⟩ = p(0) − p(1).

**X‑measurement** (same idea, different basis): project onto |+⟩,|−⟩. In practice, **H then Z**.

---

## 7) Inner products create interference

* X probabilities for |ψ⟩=α|0⟩+β|1⟩:

  * p(+)=½|α+β|², p(−)=½|α−β|².
* The cross‑term Re(α*β) is the **interference** piece (depends on relative phase). That’s why phase matters in X/Y but not in Z.

---

## 8) Mixture vs Superposition (why density matrices)

* **Pure superposition** (coherent): ρ = |ψ⟩⟨ψ| has **off‑diagonals** (coherence). Example |+⟩: ρ = ½[[1,1],[1,1]]. In X, outcome is **deterministic** (+).
* **Classical mixture** (incoherent): e.g., “50% |0⟩, 50% |1⟩” → ρ = I/2 (off‑diagonals = 0). In **any basis**, outcomes are **50–50**.
* **Subsystem of entangled pair** (local view): also mixed (often I/2). No single ket describes it.
* **Measure‑and‑forget**: measuring |+⟩ in Z then discarding the result yields ρ = I/2 (dephased).

**Rules with ρ:**

* Probabilities: p(k) = Tr(Pk ρ).
* Expectation: ⟨A⟩ = Tr(A ρ).
* Purity: Tr(ρ²)=1 (pure), <1 (mixed). Totally mixed: ρ=I/2.

---

## 9) Tiny “how to remember” checklist

* **Z** = which path / population. **X** = add vs subtract (0°). **Y** = ±90° lead/lag.
* **RY** changes 0↔1 with real amplitudes; **RX** adds ±i; **RZ** twists phase only.
* **RZZ** = phase by parity (same/different). **RXX/RYY** = pair‑swaps with phases.
* **CRZ** = RZ on target iff control=1 (ZZ‑style conditional phase).
* **Projector** = keep‑only‑this‑direction; probabilities are squared lengths of those “kept parts”.
* **Superposition vs mixture** = coherence (off‑diagonals) vs none.

---

## 10) Minimal examples (one‑liners)

* Z‑prob from |ψ⟩: p(0)=|α|², p(1)=|β|².
* X‑prob from |ψ⟩: p(+)=½|α+β|².
* RZZ(θ)|00⟩ = e^{−iθ/2}|00⟩ (phase only); on superpositions → entanglement.
* Y gate on basis: Y|0⟩=i|1⟩, Y|1⟩=−i|0⟩; RY(π) = −i·Y (same up to global phase).
* Mixture I/2 stays I/2 under any unitary: U(I/2)U† = I/2.

Keep this page handy; read top‑to‑bottom once and you’ll refresh the whole picture in minutes.

# ZZ and RZZ — Parity & Gate Action (Notes Section)

## 1) Why (Z\otimes Z) gives (+1) (even parity) and (-1) (odd parity)

Single-qubit Pauli-(Z) is
[
Z = \begin{bmatrix}1 & 0\ 0 & -1\end{bmatrix},\quad Z\lvert0\rangle = +\lvert0\rangle,; Z\lvert1\rangle = -\lvert1\rangle.
]
So (\lvert0\rangle) is a (+1) eigenstate, (\lvert1\rangle) is a (-1) eigenstate.

For two qubits, (Z\otimes Z) acts as “apply (Z) on the first qubit and (Z) on the second.” On a computational-basis state (\lvert ab\rangle) with (a,b\in{0,1}):
[
(Z\otimes Z)\lvert ab\rangle = (Z\lvert a\rangle)\otimes(Z\lvert b\rangle) = \lambda_a,\lambda_b,\lvert ab\rangle,
]
where (\lambda_0=+1), (\lambda_1=-1).

Therefore the eigenvalue is the **product** of the single-qubit signs:

* (\lvert00\rangle): (+1\times+1=+1)
* (\lvert11\rangle): ((-1)\times(-1)=+1)
* (\lvert01\rangle): (+1\times(-1)=-1)
* (\lvert10\rangle): ((-1)\times+1=-1)

“Even parity” (both equal: 00 or 11) (\Rightarrow +1).
“Odd parity” (different: 01 or 10) (\Rightarrow -1).

Matrix view (basis order (\lvert00\rangle,\lvert01\rangle,\lvert10\rangle,\lvert11\rangle)):
[
Z\otimes Z = \operatorname{diag}(+1,-1,-1,+1).
]

---

## 2) How to apply a two-qubit gate like (R_{ZZ}(\theta)) to (\lvert00\rangle)

**Definition:**
[
R_{ZZ}(\theta) = \exp!\Big(-i,\tfrac{\theta}{2}, Z\otimes Z\Big).
]

**Key trick:** ((Z\otimes Z)^2=I). For any matrix (A) with (A^2=I),
[
\mathrm{e}^{-i\alpha A} = \cos\alpha, I - i,\sin\alpha, A.
]
Here (A=Z\otimes Z), (\alpha=\theta/2). So
[
R_{ZZ}(\theta) = \cos!\Big(\tfrac{\theta}{2}\Big), (I\otimes I) ;-
; i,\sin!\Big(\tfrac{\theta}{2}\Big), (Z\otimes Z).
]

Because computational-basis states are eigenstates of (Z\otimes Z), (R_{ZZ}(\theta)) just multiplies each basis state by a phase (\mathrm{e}^{-i(\theta/2),\lambda}) where (\lambda\in{+1,-1}) is that state’s eigenvalue:

* (\lvert00\rangle) (eigenvalue (+1)):
  [ R_{ZZ}(\theta)\lvert00\rangle = \mathrm{e}^{-i\theta/2},\lvert00\rangle. ]
* (\lvert11\rangle) (eigenvalue (+1)):
  [ R_{ZZ}(\theta)\lvert11\rangle = \mathrm{e}^{-i\theta/2},\lvert11\rangle. ]
* (\lvert01\rangle) (eigenvalue (-1)):
  [ R_{ZZ}(\theta)\lvert01\rangle = \mathrm{e}^{+i\theta/2},\lvert01\rangle. ]
* (\lvert10\rangle) (eigenvalue (-1)):
  [ R_{ZZ}(\theta)\lvert10\rangle = \mathrm{e}^{+i\theta/2},\lvert10\rangle. ]

Matrix form (same basis order) makes this explicit:
[
R_{ZZ}(\theta)=\operatorname{diag}\big(\mathrm{e}^{-i\theta/2},, \mathrm{e}^{+i\theta/2},, \mathrm{e}^{+i\theta/2},, \mathrm{e}^{-i\theta/2}\big).
]

**Special note for (\lvert00\rangle):** the phase (\mathrm{e}^{-i\theta/2}) is **global** if (\lvert00\rangle) is the only component; global phase has no observable effect. The gate’s effect becomes physical when there is superposition over even and odd parity, creating a **relative phase** between them.


# Intuition: Pre-rotation → Entangler → Post-rotation (Parity Lever)

**One-line takeaway:** After entangling, the qubits are linked. A single local rotation (like `RY` on one qubit) becomes a lever on a joint property (parity), shifting probability between the **even** set `{|00⟩, |11⟩}` and the **odd** set `{|01⟩, |10⟩}`.

## Why each step exists

**Pre-rotation (e.g., `RY` on control):** creates superposition so the entangler has two branches to act on. Without it (starting from `|00⟩`), `CNOT`/`CZ` often does nothing observable.

**Entangler (`CNOT`/`CZ`/`RZZ`):** ties amplitudes across qubits (creates correlation). After `RY₀ → CNOT`, the state sits entirely in the even subspace: `cos(a)|00⟩ + sin(a)|11⟩`.

**Post-rotation (`RY` on one qubit):** a non-commuting local rotation that mixes Z-basis populations, spilling weight from even → odd. This is the **parity mixer**.

## Minimal math snapshot (for `RY₀ → CNOT → RY₁`)

Let `a = θ₁/2`, `b = θ₂/2`. After the block:

- `P₀₀ = cos²a · cos²b`, `P₀₁ = cos²a · sin²b`  
- `P₁₀ = sin²a · sin²b`, `P₁₁ = sin²a · cos²b`

**Even vs odd split:** `P_even = cos²b`, `P_odd = sin²b`  
**Expectation:** `⟨Z⊗Z⟩ = P_even − P_odd = cos(θ₂)` — independent of `θ₁`.  
Set `θ₂ = π ⇒ P_odd = 1 ⇒ ⟨ZZ⟩ = −1`.

## Key commutation fact

`RY` does **not** commute with `Z`, so it changes Z-basis probabilities (mixes `0↔1`). `RZ` **does** commute with `Z` and cannot change Z probabilities — it won’t move even↔odd weight.

## Mental models

- **“Parity rooms” + dimmer knob:** post `RY` is the door between even/odd rooms with an opening controlled by `θ₂`.
- **“Phase → probability” converter:** phase-type entanglers (`CZ`/`RZZ`) create relative phases; post `RY` turns those into Z-basis population differences you can measure.
- **“One-qubit lever on a two-qubit property”:** entanglement makes a local rotation steer a joint statistic (parity).

## Practical notes

- For **pure ZZ** minimization from `|00⟩`, you could skip entangling and prepare an odd basis state directly (e.g., `RY₁(π) → |01⟩`). We keep the full block because real Hamiltonians are **sums of terms** (`ZZ + XX +` fields) that require correlations.
- Equivalent builds: use **`CZ`** instead of `CNOT` (up to `H` on target), or **`RZZ(φ)`** sandwiched by basis changes (e.g., `H⊗H → RZZ(φ) → H⊗H`).

---

# Ansatz design — the parity lever recipe (quick notes)

**Goal:** minimize `⟨Z⊗Z⟩` by pushing probability from even `{00, 11}` to odd `{01, 10}`.

## Steps

**Pre-rotation (e.g., `RY` on control)**  
Purpose: create superposition so the entangler can act on two branches.  
Without this from `|00⟩`, `CNOT`/`CZ` often has no observable effect.

**Entangler (`CNOT`/`CZ`/`RZZ`)**  
Purpose: tie amplitudes across qubits so a single local change can steer a joint property (parity).  
After `RY` on control then `CNOT`, the state sits in the even subspace: `cos(a)|00⟩ + sin(a)|11⟩`.

**Post-rotation (`RY` or `RX` on one qubit)**  
Purpose: parity mixer; it does not commute with `Z`, so it moves weight between even and odd sets.  
This is the **single effective knob** that actually changes the even↔odd split before Z measurement.

## Minimal math snapshot (for `RY` on q0 → `CNOT` → `RY` on q1)

Let `a = θ1/2`, `b = θ2/2`.  
`P_even = cos²(b)`, `P_odd = sin²(b)`.  
`⟨Z⊗Z⟩ = P_even − P_odd = cos(θ2)`. Set `θ2 = π → ⟨ZZ⟩ = −1`.

## Mental model

Think of even/odd as two rooms; the post `RY` is the door with a dimmer knob (`θ2`) that controls how much probability flows into the odd room.
