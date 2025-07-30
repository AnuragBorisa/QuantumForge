# qGAN Architecture & Adversarial Loop

![Adversarial Loop Diagram](architecture.svg)

## Overview

A quantum GAN (qGAN) pairs a parameterized quantum‐circuit generator **G(θ)** with a classical discriminator **D(φ)**. G(θ) prepares an n-qubit state and, after many measurements, yields a probability vector **p**. D(φ) takes **p** as input and outputs a “real vs. fake” score between 0 and 1. We update φ by backpropagating the discriminator loss and update θ via the parameter-shift rule, estimating gradients by running G at θ±π/2.

## Adversarial Loop Summary

1. **Forward pass**: run G(θ), measure many times to build **p**, feed **p** into D(φ).  
2. **Loss computation**: compute discriminator loss on real vs. fake and generator loss on fake samples.  
3. **Backward pass**: update φ via standard backprop; for each θᵢ, run G at θᵢ+π/2 and θᵢ−π/2, subtract their losses, divide by 2 to get ∂L/∂θᵢ, and update θ.  

This loop trains G to produce samples that D can no longer tell apart from real data, closing the adversarial cycle.
