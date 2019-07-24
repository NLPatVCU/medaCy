from medacy.data import Dataset
from medacy.relation import Segmentation

sample_train = Dataset('/home/samantha/Desktop/Research/Data/i2b2/sample_train')
sample_test = Dataset('/home/samantha/Desktop/Research/Data/i2b2/sample_test')
training_dataset = Dataset('/home/samantha/Desktop/Research/Data/i2b2/train_data')
testing_dataset = Dataset('/home/samantha/Desktop/Research/Data/i2b2/test_data')

seg_sampleTrain = Segmentation(sample_train)
# sample_sentTrain = seg_sampleTrain.segments['sentence']
# sample_labelTrain = seg_sampleTrain.segments['label']

# seg_train = Segmentation(training_dataset)
# print(len(seg_train.segments['sentence']))
# print(len(seg_train.segments['label']))

# seg_test = Segmentation(testing_dataset)
# print(seg_test.segments['sentence'])
# print(seg_test.segments['label'])