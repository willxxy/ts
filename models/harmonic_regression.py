import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import argparse
import os
from statsmodels.formula.api import ols

def get_args():
    parser = argparse.ArgumentParser(description='Least Squares Regression Model')
    parser.add_argument('--ratio', type=float, default=0.8, help='Number of time steps')
    parser.add_argument('--dataset', type=str, help='Dataset path to apply least squares regression model')
    parser.add_argument('--verbose', action = 'store_true', default = None, help='Print verbose')
    return parser.parse_args()


def main(args):
    print('--------------------------------------------------')
    random.seed(1)
    data_path = os.path.join('./data/all_six_datasets/', args.dataset)
    df = pd.read_csv(data_path)
    column_names = df.columns.tolist()
    selected_features = random.sample(column_names[1:], 1)

    if 'electrograms' in args.dataset:
        periodicty = 1000
    else:
        df.insert(0, 'Time', range(len(df)))
        periodicty = 52
        column_names = df.columns.tolist()
        df.set_index(column_names[0], inplace=True)
    
    df['time'] = np.arange(len(df.index))
    df['sin_1'] = np.sin(2 * np.pi * df['time'] / periodicty)
    df['cos_1'] = np.cos(2 * np.pi * df['time'] / periodicty)
    df['sin_2'] = np.sin(4 * np.pi * df['time'] / periodicty)
    df['cos_2'] = np.cos(4 * np.pi * df['time'] / periodicty)
    
    train_ratio = args.ratio
    split_index = int(len(df) * train_ratio)
    if args.ratio==0:
        train_data = df
        test_data = train_data
    else:
        train_data = df.iloc[:split_index]
        test_data = df.iloc[split_index:]

    if args.verbose:
        print(column_names)
    
    formula = f'{selected_features[0]} ~ sin_1 + cos_1 + sin_2 + cos_2'
    model = ols(formula, data=train_data).fit()


    test_data['predicted'] = model.predict(test_data[['sin_1', 'cos_1', 'sin_2', 'cos_2']])

    plt.figure(figsize=(10, 5))
    plt.plot(test_data.index, test_data[selected_features[0]], label='Actual')
    plt.plot(test_data.index, test_data['predicted'], label='Predicted', linestyle='--')
    plt.title('Harmonic Regression Fit')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(f"./pngs/model_outputs/harmonic_regression_{args.dataset.split('/')[-1].split('.')[0]}.png")

    mse = mean_squared_error(test_data[selected_features[0]], test_data['predicted'])
    mae = mean_absolute_error(test_data[selected_features[0]], test_data['predicted'])
    print(f'Mean Absolute Error: {mae}')
    print(f'Mean Squared Error: {mse}')

    print('--------------------------------------------------')


if __name__ == '__main__':
    args = get_args()
    main(args)