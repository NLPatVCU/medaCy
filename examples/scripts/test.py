from sys import argv
import logging

from medacy.data import Dataset
from medacy.ner import Model
from medacy.ner.pipelines import LstmClinicalPipeline

logging.basicConfig(level=logging.INFO)

dataset = Dataset(argv[1])
labels = list(dataset.get_labels())
pipeline = LstmClinicalPipeline(entities=labels)

model = Model(pipeline)
model.fit(dataset)
model.cross_validate(num_folds=5)