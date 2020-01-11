import argparse

import numpy as np

from medacy.data.dataset import Dataset


def calculate_dataset_confusion_matrix(dataset_1_path, dataset_2_path, leniency=0.0):
    dataset_1 = Dataset(dataset_1_path)
    dataset_2 = Dataset(dataset_2_path)
    ents, mat = dataset_1.compute_confusion_matrix(dataset_2, leniency=leniency)
    return mat


def main():
    parser = argparse.ArgumentParser(description="Calculate and display the ambiguity of two datasets")
    parser.add_argument('dataset_1', type=str, help="The first dataset path")
    parser.add_argument('dataset_2', type=str, help="The second dataset path")
    parser.add_argument('-l', '--leniency', type=float, default=0.0, help="Leniency between 0.0 and 1.0 (default to 0.0)")
    args = parser.parse_args()
    result = calculate_dataset_confusion_matrix(args.dataset_1, args.dataset_2, leniency=args.leniency)
    mat = np.matrix(result)
    print(mat)


if __name__ == '__main__':
    main()
