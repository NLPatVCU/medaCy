import os
import tempfile
import shutil
import re
from medacy.data.dataset import Dataset
from medacy.tools.calculators.evaluator import evaluate_annotation_agreement


def calculate_inter_dataset_agreement(predicted_dataset: Dataset, gold_dataset: Dataset, different_entity_mappings: dict):
    """
    Evaluate the intersection of two datasets when the entity class names in the gold dataset are more specific
    than their names in the predicted dataset.
    :param predicted_dataset: The dataset to be evaluated
    :param gold_dataset: The gold dataset
    :param different_entity_mappings: A dict where a tuple of the entity type names in the gold dataset are mapped to
    their analog in the predicted dataset
    :return: The tab-delimited values
    """
    if isinstance(predicted_dataset, str) and os.path.isdir(predicted_dataset):
        predicted_dataset = Dataset(predicted_dataset)
    elif not isinstance(predicted_dataset, Dataset):
        raise TypeError("predicted_dataset must be type Dataset or a str referring to a valid directory")

    if isinstance(gold_dataset, str) and os.path.isdir(gold_dataset):
        gold_dataset = Dataset(gold_dataset)
    elif not isinstance(gold_dataset, Dataset):
        raise TypeError("gold_dataset must be type Dataset or a str referring to a valid directory")

    if len(predicted_dataset) != len(gold_dataset):
        raise ValueError("Both datasets must contain parallel copies of the same files, but the two datasets "
                         "are different lengths")

    # Unpack the tuple keys into a flattened dictionary
    mappings = {}
    for k, v in different_entity_mappings.items():
        if isinstance(k, str):
            mappings[k] = v
        if isinstance(k, tuple):
            for w in k:
                mappings[w] = v

    # Create a temp dir for the modified dataset
    temp_dataset_dir = tempfile.mkdtemp()

    # Rewrite the ann files to have the less-specific class names
    for file in [f for f in os.listdir(gold_dataset.data_directory) if f.endswith(".ann")]:
        with open(os.path.join(gold_dataset.data_directory, file)) as f:
            text = f.read()

        for k, v in mappings.items():
            k = re.escape(k)
            pattern = r"\t" + k + " "
            while re.search(pattern, text):
                text = re.sub(pattern, r"\t" + re.escape(v) + " ", text)

        with open(os.path.join(temp_dataset_dir, file), "w+") as f:
            f.write(text)

    # Pass the two directories to the evaluation calculator
    data = evaluate_annotation_agreement(predicted_dataset.data_directory, temp_dataset_dir, set(mappings.values()), relations=[])

    # Delete the temp dir
    shutil.rmtree(temp_dataset_dir)

    return data
