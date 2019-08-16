"""Converts an ANN file to tab delimited values"""

import os
from medacy.data.dataset import Dataset
from medacy.data.data_file import DataFile
from medacy.tools.entity import Entity


def ann_to_tsv(ann_file, output_dir):
    """
    Converts a single file to TSV
    :param ann_file: The path to the ann file or a DataFile object
    :param output_dir: The directory to write the outputted data
    :return: None
    """
    if isinstance(ann_file, DataFile):
        output_file_name = ann_file.file_name + ".tsv"
        ann_file_path = ann_file.ann_path
        ents = Entity.init_from_doc(ann_file_path)
    elif isinstance(ann_file, str):
        output_file_name = os.path.basename(ann_file)[:-4] + ".tsv"
        ents = Entity.init_from_doc(ann_file)
    else: raise TypeError("ann_file must be DataFile or str")

    output_file_path = os.path.join(output_dir, output_file_name)
    output_str = ""

    for e in ents:
        output_str += f"T{e.num}\t{e.ent_type}\t{e.start}\t{e.end}\t{e.text}\n"

    print(output_str)

    with open(output_file_path, "w+") as f:
        f.write(output_str)


def dataset_to_tsv(dataset, output_dir):
    if isinstance(dataset, str):
        if not os.path.isdir(dataset):
            raise NotADirectoryError("dataset was passed as a str, but the directory was not found: %s" % dataset)
        dataset = Dataset(dataset)
    elif not isinstance(dataset, Dataset):
        raise TypeError("dataset must be Dataset or str")

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    for file in dataset:
        ann_to_tsv(file, output_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert a single ANN file or a directory of ANN files to TSV files")
    parser.add_argument("input", help="The path to a single file to convert or the path to a directory to convert")
    parser.add_argument("output", help="The output directory (individual file name(s) will always be based on the input file name")
    args = parser.parse_args()

    if os.path.isfile(args.input):
        ann_to_tsv(args.input, args.output)
    elif os.path.isdir(args.input):
        dataset_to_tsv(args.input, args.output)
    else: raise ValueError("input is not an individual file or a directory")
