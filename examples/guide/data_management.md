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

Assuming your directory is structured as follows:

```
home/medacy/data
├── file_one.ann
├── file_one.txt
├── file_two.ann
└── file_two.txt
```

running:

```python
from medacy.data import Dataset
from medacy.pipeline_components import MetaMap

data = Dataset('/home/medacy/data')
for data_file in data.get_data_files():
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
['file_one.txt', file_two]
False
True
```

If all your data metamapped successfully, your directory will look like this:

```
data
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

An *medaCy compatible dataset* is python package wrapping data that can be hooked into medaCy. We can install a *medaCy compatible dataset* just like any python package. For instance,


`pip install https://github.com/NanoNLP/medaCy_dataset_end/archive/v1.0.2.tar.gz#egg=medacy_dataset_end-1.0.2`

will install `v1.0.02` of the [END](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5644562/) dataset. Alternatively,

`pip install git+https://github.com/NanoNLP/medaCy_dataset_end.git`

will install the latest version of the [END](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5644562/) dataset.

After you have installed a *medaCy compatible dataset*, loading it returns a configured `Dataset` object alongside meta-data in a `tuple` as follows:

```python
from medacy.data import Dataset

dataset, entities = Dataset.load_external('medacy_dataset_end')

```

alternatively, import the datasets package and directly call the load method:

```python
import medacy_dataset_end

dataset, entities = medacy_dataset_end.load()

```

## Using a Dataset

