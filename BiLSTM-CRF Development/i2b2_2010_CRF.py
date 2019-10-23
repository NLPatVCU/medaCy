from medacy.data import Dataset
from medacy.ner.model import Model
from medacy.ner.pipelines import SystematicReviewPipeline
import logging, sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

i2b2_2010_dataset = Dataset(data_directory="/home/conteam/i2b2_2010_concepts")

pipeline = SystematicReviewPipeline(entities=['problem', 'test', 'treatment'])

model = Model(pipeline)

model.cross_validate(training_dataset=i2b2_2010_dataset)
