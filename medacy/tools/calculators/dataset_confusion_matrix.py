import argparse
import tabulate

import numpy as np

from medacy.data.dataset import Dataset


def calculate_dataset_confusion_matrix(groundtruth_dataset_path, prediction_dataset_path, leniency=0.0, data_limit=None, density=False, heatmap=False, unmatched=False, red=False):
    groundtruth_dataset = Dataset(groundtruth_dataset_path, data_limit=data_limit)
    prediction_dataset = Dataset(prediction_dataset_path, data_limit=data_limit)
    ents, mat = groundtruth_dataset.compute_confusion_matrix(prediction_dataset, unmatched=unmatched, leniency=leniency)
    return (tabulate.tabulate(mat, headers=format_headers(ents, unmatched), showindex=ents, tablefmt="plain"))

def format_headers(entities, unmatched):
    result = []
    if unmatched:
        entities.append("Unmatched")
    for string in entities:
        vertical_string = ""
        for char in string:
            vertical_string += (char+"\n")
        result.append(vertical_string.strip())
    return result

def format_density(mat):  
    for row_index, row in enumerate(mat):
        row_total = round(float(sum(row)), 3)
        if row_total==0.0:
            for col_index, item in enumerate(row):
                mat[row_index][col_index] = 0.0
        else:
            for col_index, item in enumerate(row):
                mat[row_index][col_index] /= row_total
    return mat

def format_heatmap(mat, density):
    pass
    if(density):
        for row_index, row in enumerate(mat):
            pass
    else:
        density_mat = format_density(mat)

    pass

def main():
    parser = argparse.ArgumentParser(description="Calculate and display the ambiguity of two datasets")
    parser.add_argument('groundtruth_dataset', type=str, help="Path to the groundtruth dataset")
    parser.add_argument('prediction_dataset', type=str, help="Path to the prediction dataset")
    parser.add_argument('-l', '--leniency', type=float, default=0.0, help="Leniency to be used, must be between 0.0 and 1.0")
    parser.add_argument('-dl', '--data_limit', type=int, default=None, help="Limits the number of data files to be used in creating each dataset")
    parser.add_argument('-d', '--density', action='store_true', help="Displays values as density")
    parser.add_argument('-h', '--heatmap', action='store_true', help="Displays heatmap overlay")
    parser.add_argument('-u', '--unmatched', action='store_true', help="Displays unmatched counts")
    parser.add_argument('-r', '--red', action='store_true', help="\033[31m" + "Red?" + "\033[39m")
    args = parser.parse_args()

    print(calculate_dataset_confusion_matrix(args.groundtruth_dataset, args.prediction_dataset, args.leniency, args.data_limit, args.density, args.heatmap, args.unmatched, args.red))

if __name__ == '__main__':
    main()
