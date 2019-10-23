from medacy.data import Dataset
from medacy.ner.model import Model
from medacy.ner.pipelines import SystematicReviewPipeline
import logging, sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

n2c2_dataset = Dataset(data_directory="/home/share/N2C2")

pipeline = SystematicReviewPipeline(entities=['ADE', 'Dosage', 'Drug'])

model = Model(pipeline)

model.cross_validate(training_dataset=n2c2_dataset)
