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
from medacy.data import Dataset

data = Dataset('/home/medacy/data')
```

MedaCy **does not** alter the data you load in any way - it only reads from it.

A common data work flow might look like this.

running:

```python
from medacy.data import Dataset
from medacy.pipeline_components import MetaMap

dataset = Dataset('/home/medacy/data')
for data_file in dataset:
  print(data_file.file_name)
print(data)
print(data.is_metamapped())

metamap = Metamap('/home/path/to/metamap/binary')
data.metamap(metamap)
print(data.is_metamapped())
```

outputs:

```python
file_one
file_two
['file_one.txt', 'file_two.txt']
False
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



## Loading a medaCy compatible dataset
Using a *medaCy compatible dataset* package to manage your training data insures that data is easy and efficient to access, versioned for replicability, and distributable (selectively!).

A *medaCy compatible dataset* is python package wrapping data that can be hooked into medaCy. We can install a *medaCy compatible dataset* just like any python package. For instance,


`pip install https://github.com/NanoNLP/medaCy_dataset_end/archive/v1.0.3.tar.gz#egg=medacy_dataset_end-1.0.3`

will install `v1.0.03` of the [END](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5644562/) dataset. Alternatively,

`pip install git+https://github.com/NanoNLP/medaCy_dataset_end.git`

will install the latest version of the [END](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5644562/) dataset.

After you have installed a *medaCy compatible dataset*, loading it returns a configured `Dataset` object alongside meta-data in a `tuple` as follows:

```python
from medacy.data import Dataset

training_dataset, evaluation_dataset, meta_data = Dataset.load_external('medacy_dataset_end')

```

alternatively, import the datasets package and directly call the load method:

```python
import medacy_dataset_end

training_dataset, evaluation_dataset, meta_data = medacy_dataset_end.load()

print(meta_data['entities']) #entities this dataset annotates
print(meta_data['relations']) #relations this dataset annotates (END has None)

training_dataset = medacy_dataset_end.load_training_dataset() #access just training

evaluation_dataset = medacy_dataset_end.load_evaluation_dataset() #access just evaluation

```

## Using a Dataset
A *Dataset* is utilized for two main tasks:

1. [Model Training](#model-training)
2. [Model Prediction](#model-prediction)

### Model Training
To utilize a *Dataset* for training insure that the data you're loading is valid training data in a supported annotation format. After creating a *Model* with a processing *Pipeline*, simply pass the *Dataset* in for prediction. Here is an example of training an NER model for extraction of information relevant to nano-particles.

```python
from medacy.data import Dataset
from medacy.pipelines import FDANanoDrugLabelPipeline

dataset = Dataset('/home/medacy/data')
entities = ['Nanoparticle', 'Dose']
pipeline = FDANanoDrugLabelPipeline(entities=entities)
model = Model(pipeline, n_jobs=1)

model.fit(dataset)
```

**Note**: Unless you have tuned your *Pipeline* to extract features relevant to your problem domain, the trained model will likely not be very predictive. See [Training a model](model_training.md).

### Model Prediction

Once you have a trained or imported a model, pass in a Dataset object for bulk prediction of text.

```python
from medacy.data import Dataset
from medacy.model import Model

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

