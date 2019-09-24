from medacy.data import Dataset
from medacy.ner.model import Model
from medacy.ner.pipelines import LstmSystematicReviewPipeline
from medacy.tools import Annotations
import logging, sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

from pprint import pprint

n2c2_dataset = Dataset(data_directory="/Users/annaconte/NLPatVCU/N2C2_Data", data_limit=3)

pipeline = LstmSystematicReviewPipeline()

model = Model(pipeline)

model.fit(n2c2_dataset.get_training_data())

model.predict(n2c2_dataset.get_training_data())

model.cross_validate(num_folds=5)


print(model)






# If evaluation data is available, validate as follows:

# model.predict(testing, prediction_directory = "")
# prediction_datasets = Dataset("path")
# testing.compute_confustion_matrix(prediction_datasets)











# for file in training:
#     print(file.get_annotation_path)

#pprint(training.compute_counts())






