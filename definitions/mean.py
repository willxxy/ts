import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Simulate a time series data
np.random.seed(0)
n = 100  
t = np.arange(n)
X_t = 0.5 * t + np.random.normal(0, 1, n)  # Time series model: linear trend + noise

# Mean function: E(X_t)
def mean_function(X):
    return np.mean(X)

# Plot time series
plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)  # First subplot for the time series
plt.plot(t, X_t, label='Time Series')
overall_mean = mean_function(X_t)
plt.axhline(y=overall_mean, color='r', linestyle='--', label=f'Overall Mean: {overall_mean:.2f}')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Time Series Plot')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('./pngs/definitions/mean.png')
plt.close()
