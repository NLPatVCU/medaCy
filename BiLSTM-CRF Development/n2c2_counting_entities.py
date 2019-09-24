from pprint import pprint
from medacy.data import Dataset

n2c2_dataset = Dataset("/Users/annaconte/NLPatVCU/N2C2_Data")

n2c2_dataset_counts = n2c2_dataset.compute_counts()

pprint(n2c2_dataset_counts)

