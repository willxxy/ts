import os
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
import random
import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression

def get_args():
    parser = argparse.ArgumentParser(description='Visualize Time Series Data')
    parser.add_argument('--ratio', type=float, default=0.8, help='Number of time steps')
    parser.add_argument('--dataset', type=str, help='Dataset path to visualize')
    parser.add_argument('--verbose', action = 'store_true', default = None, help='Print verbose')
    return parser.parse_args()

def main(args):
    print('--------------------------------------------------')
    random.seed(1)
    data_path = os.path.join('./data/all_six_datasets/', args.dataset)
    df = pd.read_csv(data_path)
    column_names = df.columns.tolist()
    if args.verbose:
        print('Column names:', column_names)
        print(df.head())
        print('Length of the dataset:', len(df))

    if 'electrograms' in args.dataset:
        df['t'] = df[column_names[0]]
    else:
        df[column_names[0]] = pd.to_datetime(df[column_names[0]])
        df.sort_values(column_names[0], inplace=True)
        df['t'] = (df[column_names[0]] - df[column_names[0]].min()).dt.days

    selected_features = random.sample(column_names[1:], 1)
    
    train_ratio = args.ratio
    split_index = int(len(df) * train_ratio)
    if args.ratio==0:
        train_data = df
        test_data = train_data
    else:
        train_data = df.iloc[:split_index]
        test_data = df.iloc[split_index:]
    
    train_X = train_data['t'].values.reshape(-1, 1)
    test_X = test_data['t'].values.reshape(-1, 1)
    train_y = train_data[selected_features[0]].values
    test_y = test_data[selected_features[0]].values

    model = LinearRegression()
    model.fit(train_X, train_y)

    y_pred = model.predict(test_X)

    trend = model.coef_[0]

    if 'electrograms' in args.dataset:
        decomp = seasonal_decompose(df[selected_features[0]], period=1000, model='additive')
    else:
        decomp = seasonal_decompose(df[selected_features[0]], period=52, model='additive')
    # fig_weighted = decomp.plot()

    fig, axes = plt.subplots(ncols=1, nrows=4, sharex=True, figsize=(14, 8))
    # Plot the observed data
    axes[0].plot(decomp.observed, label='Observed', color='blue')
    axes[0].set_ylabel('Observed')
    axes[0].legend(loc='upper left')

    # Plot the trend component
    axes[1].plot(decomp.trend, label='Trend', color='red')
    axes[1].set_ylabel('Trend')
    axes[1].legend(loc='upper left')

    # Plot the seasonal component
    axes[2].plot(decomp.seasonal, label='Seasonal', color='orange')
    axes[2].set_ylabel('Seasonal')
    axes[2].legend(loc='upper left')

    # Plot the residual component
    axes[3].plot(decomp.resid, label='Residual', color='green')
    axes[3].set_ylabel('Residual')
    axes[3].legend(loc='upper left')

    axes[3].set_xlabel('Time')
    plt.tight_layout()

    plt.savefig(f"./pngs/data_analysis/seasonal_decompose_{args.dataset.split('/')[-1].split('.')[0]}.png")
    plt.close()

    z_scores = np.abs(stats.zscore(df[selected_features[0]]))

    threshold = 3
    outliers = df[column_names[0]][z_scores > threshold]

    mae = mean_absolute_error(test_y, y_pred)
    mse = mean_squared_error(test_y, y_pred)
    
    print(f'Mean Absolute Error: {mae}')
    print(f'Mean Squared Error: {mse}')
    print(f'Trend: {trend}')
    print(f'Outliers: {outliers}')
    print('z_scores:', z_scores)
    print('--------------------------------------------------')

if __name__ == '__main__':
    args = get_args()
    main(args)

