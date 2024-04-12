import os
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
import random
import pandas as pd
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Visualize Time Series Data')
    parser.add_argument('--dataset', type=str, help='Dataset path to visualize')
    parser.add_argument('--verbose', action = 'store_true', default = None, help='Print verbose')
    return parser.parse_args()

# Augmented Dickey-Fuller Test function
def adf_test(series):
    result = adfuller(series, autolag='AIC')
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))

# Rolling Variance function
def rolling_variance_test(series, window, args):
    rolling_var = series.rolling(window=window).var()
    rolling_var.plot(title='Rolling Variance (Window = {})'.format(window))
    plt.savefig(f"./pngs/data_analysis/rolling_variance_{args.dataset.split('/')[-1].split('.')[0]}.png")
    plt.close()

# Autocorrelation function analysis
def autocorrelation_analysis(series, args):
    plot_acf(series, lags=40)
    plt.title('Autocorrelation Function')
    plt.savefig(f"./pngs/data_analysis/autocorrelation_{args.dataset.split('/')[-1].split('.')[0]}.png")
    plt.close()

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
        window = 1000
        pass
    else:
        window = 52
        df[column_names[0]] = pd.to_datetime(df[column_names[0]])
    
    df.set_index(column_names[0], inplace=True)

    selected_features = random.sample(column_names[1:], 1)

    time_series = df[selected_features[0]]
    print("Performing Augmented Dickey-Fuller Test:")
    adf_test(time_series)

    print("\nChecking for Constant Variance:")
    rolling_variance_test(time_series, window=window, args = args)  # Yearly window, assuming weekly data

    print("\nAnalyzing Autocorrelation Function:")
    autocorrelation_analysis(time_series, args)


if __name__ == '__main__':  
    args= get_args()
    main(args)