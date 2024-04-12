import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Simulate multiple time series data
np.random.seed(0)
n = 100  # Number of data points
m = 10   # Number of simulations
t = np.arange(n)  # Time index
X_t = np.array([0.5 * t + np.random.normal(0, 1, n) for _ in range(m)])  # Simulate m time series

# Calculate covariance matrix for the series
def covariance_matrix(X):
    return np.cov(X, rowvar=False)  # Compute covariance matrix across columns (time points)

cov_matrix = covariance_matrix(X_t)

# Plotting
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
for i in range(m):
    plt.plot(t, X_t[i], label=f'Series {i+1}')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Multiple Time Series Plots')
plt.legend()

plt.subplot(1, 2, 2)
sns.heatmap(cov_matrix, cmap='coolwarm', cbar=True)
plt.title('Covariance Matrix')
plt.xlabel('Time Point Index')
plt.ylabel('Time Point Index')

plt.tight_layout()
plt.savefig('./pngs/definitions/covariance.png')