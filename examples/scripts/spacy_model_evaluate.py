from __future__ import unicode_literals, print_function

import plac
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
from medacy.tools import Annotations
from medacy.data import Dataset
from medacy.ner import SpacyModel

# Note: If you're using an existing model, make sure to mix in examples of
# other entity types that spaCy correctly recognized before. Otherwise, your
# model might learn the new type, but "forget" what it previously knew.
# https://explosion.ai/blog/pseudo-rehearsal-catastrophic-forgetting

@plac.annotations(
    input_dir=("Directory of ann and txt files.", "option", "i", Path),
    spacy_model_name=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    n_folds=("Number of training iterations", "option", "n", int),
)
def main(input_dir, spacy_model_name=None, n_folds=10):
    dataset = Dataset(input_dir)
    model = SpacyModel()

    model.cross_validate(
        num_folds=n_folds,
        training_dataset=dataset,
        spacy_model_name=spacy_model_name
    )

if __name__ == "__main__":
    plac.call(main)
