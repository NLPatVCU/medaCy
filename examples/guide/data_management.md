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

## Creating a Dataset
MedaCy provides two functionalities for loading data:
1. [Loading an external medaCy compatible dataset]()
2. [Loading data on your local file ]


### Loading