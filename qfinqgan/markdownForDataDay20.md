# Day 20: Processing Real Price Data & Computing Log-Returns

> **Goal:** Load actual closing prices, compute daily log-returns, and turn those returns into a simple discrete distribution you can feed into your qGAN.

---

## 1. Load the CSV of real prices
```python
# Reads a CSV file with columns ‘Date’ and ‘Close’
df = pd.read_csv(csv_path, parse_dates=['Date'])
```
- **Why?** We need the historical closing prices in date order to see how the price changed each day.

## 2. Sort by date
```python
# Ensure rows run from oldest → newest by date
df = df.sort_values('Date')
```
- **Why?** To make sure that when we compare today’s price to yesterday’s, the rows are correctly aligned.

## 3. Extract the closing prices array
```python
# Pull just the numbers into a NumPy array
prices = df['Close'].values
```
- **Why?** We only care about the price values; other columns aren’t needed for this step.

## 4. Compute daily log-returns
```python
# prices[1:] is prices on days 2…N
# prices[:-1] is prices on days 1…(N–1)
# Dividing gives the daily return factor (today/yesterday)
# Taking log makes returns additive and symmetric
log_returns = np.log(prices[1:] / prices[:-1])
```
- **Why logs?**
  - **Additivity:** log-returns add over multiple days (instead of multiply).
  - **Symmetry:** gains and losses are treated evenly.
  - **Finance models:** assume log-returns are normally distributed.

## 5. Discretise into bins
```python
# Call your Day 18 helper:
counts, bin_edges, probs = discretise(log_returns, num_bins=num_bins)
```
- **What it gives:**
  - `counts`: how many days’ returns fell into each bin.
  - `probs` : those counts normalized so they sum to 1 (a discrete probability distribution).
  - `bin_edges`: the exact numerical boundaries of each bin.

## 6. Compute bin centers for labeling
```python
# Midpoint of each slice = (left_edge + right_edge) / 2
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
```
- **Why?** To know the “representative value” of each bin when plotting or feeding into quantum circuits.

---

**Returns** `(bin_centers, probs, bin_edges)` so downstream code always knows:
1. The numeric center of each return-range (`bin_centers`),
2. The probability of landing in that range (`probs`), and
3. The raw bin boundaries (`bin_edges`).

With these in hand, you can compare real vs. synthetic distributions, feed your qGAN state preparation, or run quantum amplitude estimation on actual market data.
