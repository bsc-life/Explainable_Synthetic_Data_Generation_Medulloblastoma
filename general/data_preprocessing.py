import pandas as pd
from sklearn.preprocessing import StandardScaler
import argparse


def data_preprocessing(train_path, test_path):

    read_train = pd.read_csv(train_path, sep=',', index_col=False)
    read_test = pd.read_csv(test_path, sep=',', index_col=False)
    genes = pd.read_csv('files/input_genes.csv', index_col=False, header=None)
    genes = list(genes[0])

    read_train = read_train[genes]
    read_test = read_test[genes]

    read_train = read_train.loc[:, ~read_train.columns.str.startswith(('RPL', 'RPS'))]
    read_test = read_test.loc[:, ~read_test.columns.str.startswith(('RPL', 'RPS'))]

    scaler = StandardScaler()
    scaler.fit(read_train)
    read_train = scaler.transform(read_train)

    scaler.fit(read_test)
    read_test = scaler.transform(read_test)

    read_train = pd.DataFrame(read_train)
    read_test = pd.DataFrame(read_test)

    return read_train, read_test


def colors_preprocessing(colors_train_path, colors_test_path):

    subgroups_train = pd.read_csv(colors_train_path, sep=' ', header=None)
    colors_train = subgroups_train[1].values

    subgroups_test = pd.read_csv(colors_test_path, sep=' ', header=None)
    colors_test = subgroups_test[1].values

    return colors_train, colors_test


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # path to files
    parser.add_argument("train_path", help="path to training csv")
    parser.add_argument("test_path", help="path to test csv")

    # name of output files
    parser.add_argument("out_train_path", help="output path to training csv")
    parser.add_argument("out_test_path", help="output path to test csv")

    args = parser.parse_args()

    data_train, data_test = data_preprocessing(args.train_path, args.test_path)

    data_train.to_csv(args.out_train_path)
    data_test.to_csv(args.out_test_path)
