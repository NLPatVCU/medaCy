# Managing Data with medaCy
medaCy provides rich utilities for managing data through the
[Dataset](../../medacy/data/dataset.py) class.

In short, a [Dataset](../../medacy/data/dataset.py)
provides an abstraction from a file directory that allows other components
of medaCy to efficiently access and utilize large amounts of data. An instantiated
[Dataset](../../medacy/data/dataset.py) automatically knows its purpose
(either for prediction or training) and maintains auxillary files for
medaCy components such as Metamap accordingly.

In the context of medaCy, a [Dataset](../../medacy/data/dataset.py) is
composed of at least a collection of raw text files. Such a [Dataset](../../medacy/data/dataset.py)
is referred to as a *prediction dataset*. A [Dataset](../../medacy/data/dataset.py) can
be used for training if and only if each raw text file has a corresponding annotation file - hence,
we refer to this as a *training dataset*.

For the following examples, assume your data directory *home/medacy/data* is structure as follows:
```
home/medacy/data
├── file_one.ann
├── file_one.txt
├── file_two.ann
└── file_two.txt
```

## Table of contents
1. [Creating a Dataset](#creating-a-dataset)
2. [Using a Dataset](#using-a-dataset)

## Creating a Dataset
MedaCy provides two functionalities for loading data:
1. [Loading data from your machine](#loading-data-locally).
2. [Loading an existing medaCy compatible dataset](#loading-a-medacy-compatible-dataset).


## Loading data locally
To create a *Dataset*, simply instantiate one with a path to the directory containing your data.

```python
from medacy.data.dataset import Dataset
data = Dataset('/home/medacy/data')
```

MedaCy **does not** alter the data you load in any way - it only reads from it.

A common data work flow might look like this.

```pythonstub
>>> from medacy.data.datset import Dataset
>>> from medacy.pipeline_components.feature_overlayers.metamap.metamap import MetaMap

>>> dataset = Dataset('/home/medacy/data')
>>> for data_file in dataset:
...     data_file.file_name
'file_one'
'file_two'
>>> data
['file_one', 'file_two']
>>> data.is_metamapped()
False
>>> metamap = Metamap('/home/path/to/metamap/binary')
>>> with metamap:
...     data.metamap(metamap)
data.is_metamapped()
True
```

If all your data metamapped successfully, your data directory will look like this:

```
home/medacy/data
├── file_one.ann
├── file_one.txt
├── file_two.ann
├── file_two.txt
└── metamapped
    ├── file_one.metamapped
    └── file_two.metamapped
```

## Using a Dataset
A *Dataset* is utilized for two main tasks:

1. [Model Training](#model-training)
2. [Model Prediction](#model-prediction)

### Model Training
To utilize a *Dataset* for training insure that the data you're loading is valid training data in a supported annotation format. After creating a *Model* with a processing *Pipeline*, simply pass the *Dataset* in for prediction. Here is an example of training an NER model for extraction of information relevant to nano-particles.

```python
from medacy.data.dataset import Dataset
from medacy.model.model import Model
from medacy.pipelines import FDANanoDrugLabelPipeline

dataset = Dataset('/home/medacy/data')
entities = ['Nanoparticle', 'Dose']
pipeline = FDANanoDrugLabelPipeline(entities=entities)
model = Model(pipeline)

model.fit(dataset)
```

**Note**: Unless you have tuned your *Pipeline* to extract features relevant to your problem domain, the trained model will likely not be very predictive. See [Training a model](model_training.md).

### Model Prediction

Once you have a trained or imported a model, pass in a Dataset object for bulk prediction of text.

```python
from medacy.data.dataset import Dataset
from medacy.model.model import Model

dataset = Dataset('/home/medacy/data')
model = Model.load_external('medacy_model_clinical_notes')

model.predict(dataset)
```

By default, this creates a sub-directory in your prediction dataset named *predictions*. Assuming the file structure described previously, your directory would look like this:

```
/home/medacy/data
├── file_one.txt
├── file_two.txt
└── predictions
    ├── file_one.ann
    └── file_two.ann
```

where all files under *predictions* are the trained models predictions over your test data.
