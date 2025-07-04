# Log-Normal Sampling and Statistics

## 1. What is a Log-Normal Distribution?

* A log-normal variable \$X\$ means that \$\ln(X)\$ follows a Normal (bell-curve) distribution.
* Key properties:

  * \$X\$ is always **positive**.
  * \$X\$ is **skewed right** (a few large values stretch the tail).

## 2. Choosing \$\mu\$ and \$\sigma\$ for Sampling

1. **Pick your desired median** ("middle") value, \$M\$.
2. **Set**

   $$
     \mu = \ln(M)
   $$

   so that the distribution’s median is \$e^{\mu} = M\$.
3. **Decide how spread-out** you want your samples:

   * Smaller \$\sigma\$ → samples cluster tightly around \$M\$.
   * Larger \$\sigma\$ → samples spread more widely (fatter tails).
4. **Estimating \$\sigma\$ from a 68% range** (optional):

   * If you want 68% of samples between \$L\$ and \$H\$, solve

     $$
       \sigma = \frac{\ln(H) - \ln(L)}{2}.
     $$

## 3. Formulas for Median and Mean

* **Median** (middle):
  $\text{Median} = e^{\mu}$
* **Mean** (average):
  $\text{Mean} = e^{\mu + \tfrac12\sigma^2} = (\text{Median})\times e^{\sigma^2/2}$

## 4. Example Table

| \$\sigma\$ | Median (\$=100\$) | Mean (\$=100\times e^{\sigma^2/2}\$) |
| :--------: | :---------------: | :----------------------------------: |
|     0.1    |        100        |  \$100\times e^{0.005}\approx100.5\$ |
|     0.5    |        100        |  \$100\times e^{0.125}\approx113.3\$ |
|     1.0    |        100        |   \$100\times e^{0.5}\approx164.9\$  |

## 5. Quick Quiz

* **Given** \$\mu = \ln(80)\$, \$\sigma = 0.5\$:

  * Median = \$e^{\ln(80)} = 80\$
  * Mean   = \$80\times e^{0.5^2/2} = 80\times e^{0.125}\approx90.65\$

## 6. How to Sample in Python

```python
import numpy as np

mu    = np.log(100)    # median = 100
sigma = 0.2            # control spread
samples = np.random.lognormal(mean=mu, sigma=sigma, size=10000)
```

*Keep \$\mu\$ and \$\sigma\$ simple: \$\mu\$ fixes the middle, \$\sigma\$ fixes how "bumpy" the values are.*
