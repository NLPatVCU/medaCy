from __future__ import unicode_literals, print_function

import plac
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
from medacy.tools import Annotations
from medacy.data import Dataset
from medacy.ner import SpacyModel

from sys import exit

# Note: If you're using an existing model, make sure to mix in examples of
# other entity types that spaCy correctly recognized before. Otherwise, your
# model might learn the new type, but "forget" what it previously knew.
# https://explosion.ai/blog/pseudo-rehearsal-catastrophic-forgetting

@plac.annotations(
    spacy_model_path=("Path to spaCy model to load", "option", "m", Path),
    input_dir=("Directory of ann and txt files to predict for", "option", "i", Path),
)
def main(spacy_model_path, input_dir):
    dataset = Dataset(input_dir)
    model = SpacyModel()
    model.load(spacy_model_path)
    model.predict(dataset)

if __name__ == "__main__":
    plac.call(main)
