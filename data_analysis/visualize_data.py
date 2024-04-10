import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser(description='Visualize Time Series Data')
    parser.add_argument('--dataset', type=str, help='Dataset path to visualize')
    parser.add_argument('--features', type=str, default = None, help='Feature to visualize')
    return parser.parse_args()

def visualize_data(data_path, features = None):
    df = pd.read_csv(data_path)
    column_names = df.columns.tolist()
    print('Column names:', column_names)

    if features == None:
        features = column_names[1]

    print(df.head())
    print('Length of the dataset:', len(df))

    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    df[features].plot(figsize=(10, 5))
    plt.title('Time Series Plot')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.savefig(f"./pngs/time_series_plot_{args.dataset.split('/')[-1].split('.')[0]}.png")

def main(args):
    data_path = os.path.join('./data/all_six_datasets/', args.dataset)
    visualize_data(data_path, args.features)

if __name__ == '__main__':
    args = get_args()
    main(args)