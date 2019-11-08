def scores_and_counts_to_csv(data, dataset, output_path):
    """
    Generates comma separated values for the cross-validation scores and tag counts for a given dataset
    :param data: The data outputted by model.cross_validate()
    :param dataset: The dataset used to obtain those scores
    :param output_path: where to write the output to
    :return: The string of the outputted csv
    """
    counts = dataset.compute_counts()['entities']

    pass

