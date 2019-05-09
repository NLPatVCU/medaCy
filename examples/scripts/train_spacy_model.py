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
    spacy_model_name=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    new_model_name=("New model name for model meta.", "option", "nm", str),
    input_dir=("Directory of ann and txt files.", "option", "i", Path),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(spacy_model_name=None, new_model_name="animal", input_dir=None, output_dir=None, n_iter=30):
    dataset = Dataset(input_dir)

    model = SpacyModel()
    model.fit(
        dataset = dataset,
        spacy_model_name = spacy_model_name,
        new_model_name = new_model_name,
        output_dir = output_dir
    )


if __name__ == "__main__":
    plac.call(main)
