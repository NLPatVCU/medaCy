"""Example script for evaluating spaCy models.
"""
from pathlib import Path
from datetime import datetime
import time
import logging
import plac
from medacy.data import Dataset
from medacy.ner import SpacyModel

# Note: If you're using an existing model, make sure to mix in examples of
# other entity types that spaCy correctly recognized before. Otherwise, your
# model might learn the new type, but "forget" what it previously knew.
# https://explosion.ai/blog/pseudo-rehearsal-catastrophic-forgetting

@plac.annotations(
    input_dir=("Directory of ann and txt files.", "option", "i", Path),
    spacy_model_name=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    n_folds=("Number of fold to cross validate across", "option", "n", int),
    fit_iterations=("Number of training iterations per fold", "option", "mn", int)
)
def main(input_dir, spacy_model_name=None, n_folds=10, fit_iterations=30):
    """Main function.
    """
    dataset = Dataset(input_dir)
    model = SpacyModel()

    current_time = datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H.%M.%S')
    log_path = str(input_dir) + '/build_%s.log' % current_time
    logging.basicConfig(filename=log_path, level=logging.INFO)

    model.cross_validate(
        num_folds=n_folds,
        training_dataset=dataset,
        spacy_model_name=spacy_model_name,
        iterations=fit_iterations
    )

if __name__ == "__main__":
    plac.call(main)
