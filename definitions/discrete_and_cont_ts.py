import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Discrete Time Series Example: A simple daily count series
np.random.seed(0)
date_range = pd.date_range(start='2022-01-01', periods=100, freq='D')
discrete_data = np.random.poisson(lam=5, size=len(date_range))
discrete_time_series = pd.Series(discrete_data, index=date_range)

# Continuous Time Series Example: A sine wave with noise
time = np.linspace(0, 4 * np.pi, 1000) # continuous time scale
continuous_data = np.sin(time) + np.random.normal(scale=0.5, size=len(time))
continuous_time_series = pd.Series(continuous_data, index=time)

# Plotting both time series
fig, ax = plt.subplots(2, 1, figsize=(12, 8))

# Discrete time series plot
ax[0].stem(discrete_time_series.index, discrete_time_series, 'b', markerfmt='bo', basefmt=" ")
ax[0].set_title('Discrete Time Series Example')
ax[0].set_xlabel('Date')
ax[0].set_ylabel('Count')
ax[0].grid(True)

# Continuous time series plot
ax[1].plot(continuous_time_series.index, continuous_time_series, 'r')
ax[1].set_title('Continuous Time Series Example')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Value')
ax[1].grid(True)

# Show plots
plt.tight_layout()
plt.savefig('./pngs/discrete_and_cont_ts.png')
