import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import argparse
import os

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
    column_names = df.columns.tolist() # Exclude the date column
    df[column_names[0]] = pd.to_datetime(df[column_names[0]])
    df.sort_values(column_names[0], inplace=True)
    selected_features = random.sample(column_names[1:], 1)
    
    df[column_names[0]] = pd.to_datetime(df[column_names[0]])
    df['t'] = (df[column_names[0]] - df[column_names[0]].min()).dt.days

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

    # Convert dates to a numerical format (e.g., number of days since the start of the dataset)
    train_X = train_data['t'].values.reshape(-1, 1)
    train_y = train_data[selected_features[0]].values

    test_X = test_data['t'].values.reshape(-1, 1)
    test_y = test_data[selected_features[0]].values

    # Applying a polynomial transformation for quadratic fit (degree = 2)
    poly = PolynomialFeatures(degree=2)
    train_X_poly = poly.fit_transform(train_X)
    test_X_poly = poly.fit_transform(test_X)

    # Least squares regression
    model = LinearRegression()
    model.fit(train_X_poly, train_y)

    # Predictions for the regression line
    y_pred = model.predict(test_X_poly)

    # Coefficients
    coefficients = model.coef_
    intercept = model.intercept_
    if args.verbose:
        print(f'Coefficients: {coefficients}')
        print(f'Intercept: {intercept}')

    # Plotting the data and the regression line
    plt.figure(figsize=(10, 6))
    plt.scatter(test_X, test_y, color='blue', label='Data', s= 10)
    plt.plot(test_X, y_pred, color='red', label='Least Squares Regression')
    plt.xlabel('Days since start')
    plt.ylabel(f'{selected_features[0]}')
    plt.title('Least Squares Regression')
    plt.legend()
    plt.savefig(f"./pngs/model_outputs/least_squares_regression_{args.ratio}_{args.dataset.split('/')[-1].split('.')[0]}.png")
    plt.close()

    mae = mean_absolute_error(test_y, y_pred)
    mse = mean_squared_error(test_y, y_pred)
    print(f'Mean Absolute Error: {mae}')
    print(f'Mean Squared Error: {mse}')

    print('--------------------------------------------------')


if __name__ == '__main__':
    args = get_args()
    main(args)