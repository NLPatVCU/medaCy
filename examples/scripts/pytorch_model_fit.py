"""
Script for training Pytorch models.
"""
from __future__ import unicode_literals, print_function

from pathlib import Path
from datetime import datetime
import time
import logging
import plac
from medacy.data import Dataset
from medacy.ner import PytorchModel

# Note: If you're using an existing model, make sure to mix in examples of
# other entity types that spaCy correctly recognized before. Otherwise, your
# model might learn the new type, but "forget" what it previously knew.
# https://explosion.ai/blog/pseudo-rehearsal-catastrophic-forgetting

@plac.annotations(
    input_dir=("Directory of ann and txt files.", "option", "i", Path),
    model_name=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("New model name for model meta.", "option", "o", str),
    folds=("Folds for cross validation", "option", "f", int),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(input_dir, model_name=None, output_dir=None, folds=5, n_iter=30):
    """Main function."""
    logging.basicConfig(filename='pytorch.log', level=logging.INFO)
    # logging.basicConfig(level=logging.INFO)

    start_time = time.time()
    current_time = datetime.fromtimestamp(start_time).strftime('%Y_%m_%d_%H.%M.%S')
    logging.info('START TIME: ' + current_time)

    dataset = Dataset(input_dir)
    model = PytorchModel(True)

    # model.fit(dataset, n_iter)
    # model.save()

    # model.load('second-try.pt')
    # model.evaluate_prediction(dataset)
    model.cross_validate(dataset, folds, n_iter)
    
    end_time = time.time()
    current_time = datetime.fromtimestamp(end_time).strftime('%Y_%m_%d_%H.%M.%S')
    logging.info('END TIME: ' + current_time)

    seconds_elapsed = end_time - start_time
    minutes_elapsed = seconds_elapsed / 60.0
    hours_elapsed = minutes_elapsed / 60.0
    logging.info('HOURS ELAPSED: %.0f:%.02f' % (hours_elapsed, minutes_elapsed))

if __name__ == "__main__":
    plac.call(main)
