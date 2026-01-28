

<div align="center">

# 🚦 Beyond the Adjacency Matrix: A Deep Dive into PEMS

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![NumPy](https://img.shields.io/badge/Library-NumPy%20%7C%20Pandas-orange?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Part%201%20Completed-success?style=for-the-badge)


# 📡 Part 1: focuses on **Data Profiling**
</div>


---


> **"Everyone runs baselines on PEMS08, but few stop to look at the raw signals. Are we learning physics, or are we just overfitting to noise?"**

This subject is not just another collection of baselines. It is a **first-principles investigation** into the physics, statistics, and causal structures hidden within the PEMS08 traffic dataset. Before applying complex models, we perform a critical "health check" on the data.

### Part 1 focuses on **Data Profiling**: Understanding the tensor structure, statistical distributions, and physical effectiveness of the sensor network.

### 🔬 Analysis 1: The 3D Tensor Structure

Traffic data is often regarded as a 2D matrix (Time × Nodes), but PEMS08 is fundamentally a **3-Dimensional Tensor**.

#### 🐍 Code Analysis

We load the `.npz` file to inspect its raw dimensions.

```python
import numpy as np

# Load the PEMS08 dataset
raw = np.load('data/PEMS08.npz')
data = raw['data']

print(f"Data Shape: {data.shape}")
# Output: (17856, 170, 3)

```

#### 🧠 Deep Insight

The shape `(17856, 170, 3)` reveals three critical dimensions:

1. **Time ():** Represents 62 days of continuous monitoring sampled at 5-minute intervals.
2. **Space ():** Represents 170 distinct sensors on the San Bernardino highway network.
3. **Features ():**
* **Index 0: Flow (Volume)** - The number of cars.
* **Index 1: Occupancy** - The ratio of time the sensor is occupied (0.0 to 1.0).
* **Index 2: Speed** - Average speed (mph).



> **Why this matters:** Most baselines only predict "Flow". However, **Occupancy** is physically coupled with Flow (via the Fundamental Diagram of Traffic). Ignoring the other two dimensions loses critical context about congestion states.

---

### 📊 Analysis 2: The Non-Gaussian Reality

A common pitfall in traffic forecasting is assuming the data follows a Gaussian (Normal) distribution. Our statistical profiling proves this assumption is **fundamentally wrong**.

#### 🐍 Code Analysis

We flatten the tensor to calculate global statistics, focusing on **Skewness** and **Kurtosis**.

```python
from scipy import stats
import pandas as pd

feat_labels = ['Flow', 'Occupancy', 'Speed']
stats_list = []

for i in range(3):
    feat = data[:, :, i].flatten()
    stats_list.append({
        'Feature': feat_labels[i],
        'Mean': np.mean(feat),
        'Std': np.std(feat),
        'Skewness': stats.skew(feat),
        'Kurtosis': stats.kurtosis(feat)
    })

df_stats = pd.DataFrame(stats_list).set_index('Feature')
print(df_stats)

```
#### 📈 Statistical Report

| Feature | Min | Max | Mean | Std | Diagnosis |
| --- | --- | --- | --- | --- | --- |
| **Flow** | 0.00 | 1147.00 | **230.68** | **146.22** | **Right-Skewed** |
| **Occupancy** | 0.00 | 0.90 | **0.07** | 0.05 | **Heavy-Tailed** |
| **Speed** | 3.00 | 82.30 | **63.76** | 6.65 | **Left-Skewed** |

#### 🧠 Deep Insight

* **Flow (Mean 230, Max 1147):** The distribution has a massive range. The high variance implies that predicting peak hour traffic (near 1147) is significantly harder than predicting the mean.
* **Occupancy (Mean 0.07):** This implies sparsity—most of the time, the road occupancy is very low (7%), but it can spike to 90% during jams.
* **Speed (Mean 63.76 mph):** This confirms the sensors are on a highway (San Bernardino), where free-flow speed is around 65 mph.

> **Takeaway:** Using standard **MSE (Mean Squared Error)** is risky because it is sensitive to outliers. The heavy-tailed nature suggests that **Huber Loss** or **MAE** might be more robust for training stable models.

<p align="center">
<img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/fea7bba6-77fe-46f7-a9d9-e587c3411b4f" />


<em>Figure 1: The S-shaped Q-Q plot (right) confirms that Traffic Flow is NOT Gaussian.</em>
</p>

### 🛠 Analysis 3: The "Zero" Ambiguity & Dead Nodes

In PeMS, a value of `0` is ambiguous. Does it mean "no cars" (e.g., at 3:00 AM) or "broken sensor" (Sensor Failure)? Distinguishing these is critical for model robustness.

#### 🐍 Code Analysis

We scan for **Dead Nodes**: sensors that return `0` for more than 99% of the timestamps.

```python
# Calculate the zero-rate for each sensor
flow_data = data[:, :, 0]
node_zero_rates = np.sum(flow_data == 0, axis=0) / flow_data.shape[0]

# Identify Dead Nodes (>99% zeros)
dead_nodes = np.where(node_zero_rates > 0.99)[0]
print(f"Dead Nodes (Always 0): {dead_nodes}")

```

#### 🧠 Deep Insight

By visualizing the **Spatio-Temporal Availability Matrix** (Figure 2), we can visually distinguish valid data from anomalies using a high-contrast color scheme:

1. **Night Zeros (<span style="color:#2980B9">Blue</span>):** Zeros appearing between 00:00 - 04:00. These are physically valid empty roads and should be learned by the model.
2. **Daytime Zeros (<span style="color:#FF0000">Red</span>):** Zeros appearing during peak hours (08:00 - 18:00). These are **Anomalies** (Sensor Failures). The vertical red streaks indicate sensors that went offline for days or weeks.
3. **Dead Nodes:** Specific nodes identified by the code that are statistically "dead". These nodes essentially inject pure noise into Graph Convolutional Networks, misleading their neighbors.

> **Takeaway:** **Graph Pruning** or **Masking** is strictly required. A robust model must mask out the <span style="color:#FF0000">**Red**</span> regions (Failures) while preserving the <span style="color:#2980B9">**Blue**</span> regions (Valid Zero Traffic).

<p align="center">
<img width="2384" height="584" alt="image" src="https://github.com/user-attachments/assets/37b47ee7-e2c6-4039-8269-5d25bab688ce" />
    
<em>Figure 2: Spatio-Temporal Availability. Vertical <b>Red Streaks</b> indicate persistent sensor failures (Anomalies), while <b>Blue</b> represents valid low-traffic periods.</em>
</p>




### 📂 Project Structure

We follow a minimalist structure to ensure immediate reproducibility.

```text
PeMS-Analysis/
├── data/
│   ├── PEMS08.npz          # Raw Traffic Tensor
│   └── PEMS08.csv          # Static Graph Topology
├── notebooks/
│   └── 01_Data_Profiling.ipynb  # [Core] Statistical Analysis & Visualization
├── images/                 # Generated Figures
│   ├── 01_distribution.png
│   └── 01_missing_matrix.png
├── requirements.txt        # Dependencies
└── README.md               # You are here

```


<div align="center">



# 📡 Part 2: Signal Processing & Spectral Analysis
</div>

---


> **"Traffic is a low-frequency signal buried in high-frequency noise. If you train on raw data, you are mostly learning noise."**

In Part 1, we analyzed the statistical properties. In **Part 2**, we shift our perspective from the **Time Domain** to the **Frequency Domain**. By applying **Fast Fourier Transform (FFT)**, we prove that traffic flow is not random; it has a distinct "heartbeat"—a dominant 24-hour cycle.

This analysis helps us understand why simple models often fail to capture the underlying physics of traffic.

---

### 📡 Analysis 1: The Frequency Spectrum & Signal Decomposition

In this section, we shift our perspective from the **Time Domain** to the **Frequency Domain**. By applying **Fast Fourier Transform (FFT)** and **Spectral Filtering**, we unveil the hidden "heartbeat" of the city and separate the true traffic trend from random noise.

#### 1. The City's "Heartbeat" (Frequency Spectrum)

We first transform the time-series signal $x(t)$ into the frequency domain $X(f)$ to identify dominant periodicities.

> **Figure 3** below visualizes the **Frequency Spectrum** of Node 100. The X-axis represents the Period (Log Scale), and the Y-axis represents Amplitude.

<p align="center">
  
  <img width="1184" height="584" alt="image" src="https://github.com/user-attachments/assets/b259bea4-f31d-4f80-b6c7-13252d022479" />
  <br>
  <em><b>Figure 3:</b> The Frequency Spectrum reveals the "Heartbeat" of traffic flow.</em>
</p>

**🧠 Deep Insight: The Harmonic Structure**
The spectrum is not random; it shows three distinct, mathematically precise peaks:
* <span style="color:#e74c3c">**24h (Fundamental Frequency):**</span> The dominant spike. It represents the **Circadian Rhythm** of human society—the daily cycle of waking up, working, and sleeping. This is the strongest physical force driving traffic.
* <span style="color:#e74c3c">**12h (2nd Harmonic):**</span> This peak captures the **Double-Hump Structure** of the day (Morning Rush + Evening Rush). A single 24h sine wave cannot represent two peaks; the 12h component adds this detail.
* <span style="color:#e74c3c">**8h (3rd Harmonic):**</span> This component fine-tunes the shape of the waveform, representing the "Off-Peak" or "Inter-Peak" transitions.

> **Takeaway:** Traffic flow is a superposition of these strong periodic signals. A model without explicit **Time-of-Day Embeddings** or **Periodic Inductive Bias** will struggle to capture this fundamental physics.

---

#### 2. Separating Trend from Noise (Spectral Decomposition)

Raw traffic data is noisy. Based on the spectrum above, we apply a **Low-Pass Filter** to keep the dominant cycles (>4h) and remove high-frequency noise.

> **Figure 4** demonstrates this decomposition. We reconstruct the signal using Inverse FFT (iRFFT).

<p align="center">
  <img width="1483" height="784" alt="image" src="https://github.com/user-attachments/assets/ca20ac02-6f6f-4ec2-97f2-68ac39fb70a0" />
  
  <br>
  <em><b>Figure 4:</b> (Top) The extracted <b>Trend (Orange)</b> vs. Raw Signal. (Bottom) The residual <b>High-Frequency Noise (Blue)</b>.</em>
</p>

**🧠 Deep Insight: Epistemic vs. Aleatoric Uncertainty**
* **The Trend (Orange Line):** This captures the **Predictable** part of traffic (the daily commute patterns). A good model should overfit to this line.
* **The Noise (Blue Area):** This captures the **Unpredictable** (Aleatoric) uncertainty, such as random braking, sensor jitter, or minor accidents.
* **The Trap:** If you train a model (like a vanilla LSTM) on the raw data with standard MSE loss, it often wastes capacity trying to predict the "Blue Noise," leading to poor generalization.



---

### 📉 Analysis 2: Trend vs. Noise Decomposition

Raw traffic data is noisy. By applying a **Low-Pass Filter** in the frequency domain, we can separate the "True Trend" from "Random Noise".

#### 🐍 Code Analysis

We zero out high-frequency components (Noise) and reconstruct the signal using Inverse FFT (`irfft`).

```python
from scipy.fft import irfft

# Define Cutoff: Keep only frequencies lower than 1 cycle per 2 hours
cutoff_freq = 1 / 2.0 
yf_clean = yf.copy()
yf_clean[xf > cutoff_freq] = 0

# Reconstruct
clean_signal = irfft(yf_clean)
noise = signal - clean_signal

```

#### 🧠 Deep Insight

* **Trend (Low Frequency):** Represents the macro traffic pattern (e.g., rush hour formation). This is predictable.
* **Noise (High Frequency):** Represents random events (e.g., a car braking suddenly, sensor jitter). This is **unpredictable**.
* **Takeaway:** Effective forecasting models should focus on learning the trend while being robust to the high-frequency noise floor.

<p align="center">
<img src="images/02_trend_decomposition.png" width="90%">





<em>Figure 4: Signal Decomposition. The Orange line (Trend) captures the commute, while the Blue area (Residuals) captures random noise.</em>
</p>

---

### 📉 Analysis 3: Stationarity Test (ADF)

Most Time-Series models (like ARIMA) assume the data is **Stationary** (mean and variance do not change over time). Is PeMS stationary?

#### 🐍 Code Analysis

We perform the **Augmented Dickey-Fuller (ADF)** test.

```python
from statsmodels.tsa.stattools import adfuller

result = adfuller(signal[:2000]) # Test on a subset
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")

```

#### 🧠 Deep Insight

* **Result:** The p-value is often very low (), technically rejecting the null hypothesis of a unit root.
* **Nuance:** However, traffic is **Cyclostationary**, not strictly stationary. The mean shifts drastically between 3:00 AM and 5:00 PM.
* **Takeaway:** Simple normalization techniques (like standard Z-Score) might be insufficient. Advanced handling of distribution shifts (e.g., periodic normalization) is often required.

---

### 🚀 Conclusion for Part 2

We have unlocked the frequency domain secrets of PEMS08:

1. **Dominant Periodicity:** 24-hour cycle is the "Ground Truth".
2. **Spectral Sparsity:** Most information is concentrated in a few low frequencies.
3. **Noise Floor:** A significant portion of the signal is high-frequency noise.

**👉 Next Step: [Part 3 - Causal Discovery]**
Now that we understand the *Signal*, we must understand the *Structure*. We will challenge the static adjacency matrix and discover real **Time-Lagged Causal Links**.

---

<div align="center">







