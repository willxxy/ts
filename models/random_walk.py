import random
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import os
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description='Random Walk Model')
    parser.add_argument('--ratio', type=float, default=0.8, help='Number of time steps')
    parser.add_argument('--dataset', type=str, help='Dataset path to apply random walk model')
    parser.add_argument('--features', type=int, default = None, help='# of features to apply random walk model')
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

    if args.verbose:
        print(column_names)
    
    assert args.features <= len(column_names[1:]), f"Number of features should be less than or equal to {len(column_names)}"

    selected_features = random.sample(column_names[1:], args.features)
    
    # Prepare data for random walk predictions
    train_ratio = args.ratio
    split_index = int(len(df) * train_ratio)
    train_data = df.iloc[:split_index]
    test_data = df.iloc[split_index:]

    future_steps = len(test_data)

    if len(selected_features) == 1:
        print(f"Doing univariate random walk on {args.dataset.split('/')[-1].split('.')[0]}")
        sampled_feature = selected_features[0]
        train_last_value = train_data[sampled_feature].iloc[-1]
        predictions = train_last_value + np.cumsum(np.random.normal(0, train_data[sampled_feature].std(), future_steps))
        label = 'Univariate Predictions'
    else:
        print(f"Doing multivariate random walk {args.dataset.split('/')[-1].split('.')[0]}")
        train_last_values_multivariate = train_data[selected_features].iloc[-1].values
        predictions = np.expand_dims(train_last_values_multivariate, axis=0) + np.cumsum(
        np.random.multivariate_normal(np.zeros(len(selected_features)), 
                                    np.cov(train_data[selected_features].T), 
                                    future_steps), axis=0)
        sampled_feature = str(random.choice(selected_features))
        sampled_feature_index = selected_features.index(sampled_feature)
        predictions = predictions[:,sampled_feature_index]
        label = 'Multivariate Predictions'

    plt.figure(figsize=(12,6))
    plt.plot(df[column_names[0]], df[sampled_feature], label='Full Data', color='lightgrey')
    plt.plot(test_data[column_names[0]], test_data[sampled_feature], label='Ground Truth', color='blue')#, marker='o')
    plt.plot(test_data[column_names[0]], predictions, label=label, color='orange')#, marker='x')
    plt.title('Random Walk Predictions vs Ground Truth (Test Set)')
    plt.xlabel('Date')
    plt.ylabel(f'Sampled Feature: {sampled_feature}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./pngs/model_outputs/random_walk_{args.ratio}_{args.features}_{args.dataset.split('/')[-1].split('.')[0]}.png")
    plt.close()
    
    mse = mean_squared_error(test_data[sampled_feature], predictions)
    mae = mean_absolute_error(test_data[sampled_feature], predictions)
    print('Mean Squared Error:', mse)
    print('Mean Absolute Error:', mae)
    print('--------------------------------------------------')

if __name__ == '__main__':
    args = get_args()
    main(args)
