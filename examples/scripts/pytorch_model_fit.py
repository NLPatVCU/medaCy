"""
Script for training Pytorch models.
"""
from __future__ import unicode_literals, print_function

from pathlib import Path
import plac
from medacy.data import Dataset
from medacy.ner import PytorchModel

# Note: If you're using an existing model, make sure to mix in examples of
# other entity types that spaCy correctly recognized before. Otherwise, your
# model might learn the new type, but "forget" what it previously knew.
# https://explosion.ai/blog/pseudo-rehearsal-catastrophic-forgetting

@plac.annotations(
    input_dir=("Directory of ann and txt files.", "option", "i", Path),
    spacy_model_name=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("New model name for model meta.", "option", "o", str),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(input_dir, spacy_model_name=None, output_dir=None, n_iter=30):
    """Main function."""
    dataset = Dataset(input_dir)
    model = PytorchModel()
    model.fit(dataset, n_iter)

    # model.fit(
    #     dataset=dataset,
    #     spacy_model_name=spacy_model_name,
    #     iterations=n_iter,
    # )

    # if output_dir is not None:
    #     model.save(output_dir)

if __name__ == "__main__":
    plac.call(main)
