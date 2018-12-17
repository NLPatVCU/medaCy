[![spaCy](https://img.shields.io/badge/built%20with-spaCy-09a3d5.svg)](https://spacy.io)
# medaCy
:hospital: Medical Natural Language Processing with spaCy :hospital:

MedaCy is a text processing and learning framework built over spaCy to support the lightning fast prototyping, building, and application of highly predictive medical named entity
recognition systems. It is designed to streamline researcher workflow by providing utilities for model training, prediction and organization while insuring the replicability of models

![alt text](https://nlp.cs.vcu.edu/images/Edit_NanomedicineDatabase.png "Nanoinformatics")


# :star2: Features
- Highly predictive out-of-the-box trained models for clinical named entity recognition and relationship extraction.
- Customizable feature extraction pipelines for custom model building.
- Integrated converters for common text annotation formats (Prodigy, BRAT, etc).
- Pre-compiled medical terminology and abbreviation lexicons.

## :thought_balloon: Where to ask questions

MedaCy actively maintained by  [@AndriyMulyar](https://github.com/AndriyMulyar>)
and [@CoreySutphin](https://github.com/CoreySutphin). The best way to
receive immediate responses to any questions is to raise an issue.

## :computer: Installation Instructions
Medacy can be installed for general use or for pipeline development / research purpose.

| Application | Run           |
| ----------- |:-------------:|
| Prediction and Model Training with Pre-Built Pipelines (stable) | `pip install git+https://github.com/NanoNLP/medaCy.git` |
| Prediction and Model Training with Pre-Built Pipelines (latest) | `pip install git+https://github.com/NanoNLP/medaCy.git@development` |
| Pipeline Development and Contribution  | [See Contribution Instructions](/CONTRIBUTING.md) |

To install this repository from source do the following:
1) Enter into a python3 virtual envirorment, once inside make sure to upgrade pip to the latest version.
2) Run the following instruction - this should take a bit and may throw some non-fatal warnings.
```python
pip install git+https://github.com/NanoNLP/medaCy.git
```
3) Install spaCy's small model.
```python
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.0.0/en_core_web_sm-2.0.0.tar.gz
```


# :books: User Guide
Using medaCy is simple: 
1. Select a pipeline or build your own.
2. Load training/testing data (currently only BRAT annotation format is support - help us write converters!)
3. Instantiate a Model with your chosen pipeline, train on your annotated data, and retrieve a model for prediction! 

Training and using a Named Entity Recognition model for Clinical Text using medaCy:

```python
from medacy.model import Model
from medacy.pipelines import ClinicalPipeline
from medacy.tools import DataLoader
from medacy.pipeline_components import MetaMap
import logging, sys

# See what medaCy is doing at any part of the learning or prediction process
logging.basicConfig(stream=sys.stdout,level=logging.INFO) #set level=logging.DEBUG for more information

# Load in and organize traiing and testing files
train_loader = DataLoader("/training/directory")
test_loader = DataLoader("/evaluation/directory")

# MetaMap is required for powerful ClinicalPipeline performance, configure to your MetaMap path
metamap = MetaMap(metamap_path="/home/share/programs/metamap/2016/public_mm/bin/metamap")

# Optionally pre-MetaMap data to speed up performance
train_loader.metamap(metamap)
test_loader.metamap(metamap)

# Choose which pipeline to use and what entities to classify
pipeline = ClinicalPipeline(metamap, entities=['Drug', 'Form', 'Route', 'ADE', 'Reason', 'Frequency', 'Duration', 'Dosage', 'Strength'])

# Initialize a Model with the pipeline it will use to preprocess the data
# The algorithm used for prediction is specified in the pipeline - ClinicalPipeline uses CRF(Conditional Random Field)
model = Model(pipeline)

#  Run training docs through pipeline and fit the model
model.fit(train_loader) 

# Perform 10-fold stratified cross-validation on the data used to fit the model
# Can also pass in a DataLoader instance to instead cross validate on new data
model.cross_validate(num_folds=10) 

# Predictions appear in a /predictions subdirectory of your test data
model.predict(test_loader) 

```

One can also dump fitted models into a specified directory.
```python
model.fit(train_loader)
model.dump('/path/to/dump/to') # Trained model is now stored at specified directory

``` 

Note, the ClinicalPipeline requires spaCy's small model - install it with pip:
```python
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.0.0/en_core_web_sm-2.0.0.tar.gz
```


How medaCy works
================
MedaCy leverages the text-processing power of spaCy with state-of-the-art research tools and techniques in medical named entity recognition.
MedaCy consists of a set of lightning-fast pipelines that are specialized for learning specific types of medical entities. A pipeline consists
of a stackable and interchangeable set of PipelineComponents - these are bite-sized code blocks that each overlay a feature onto the text being processed.

Components
==========
You can write your own PipelineComponents to utilize in custom pipelines by interfacing the BasePipeline and BaseComponent classes. Alternatively
use the components already included with medaCy. Some more powerful components require outside software - an example is the MetaMapComponent which interfaces with MetaMap
to overlay rich medical concept information onto text. Components are chained or stacked in pipelines and can themselves depend on the outputs of previous components to function.




Reference
=========

> @ARTICLE {,
>     author  = "Andriy Mulyar, Natassja Lewinski and Bridget McInnes",
>     title   = "TAC SRIE 2018: Extracting Systematic Review Information with MedaCy",
>     journal = "National Institute of Standards and Technology (NIST) 2018 Systematic Review Information Extraction (SRIE) > Text Analysis Conference",
>     year    = "2018",
>     month   = "nov"
> }

License
=======
This package is licensed under the GNU General Public License


Authors
=======
Andriy Mulyar, Corey Sutphin, Bobby Best, Steele Farnsworth, and Bridget McInnes

Acknowledgments
===============
- [VCU Natural Language Processing Lab](https://nlp.cs.vcu.edu/)     ![alt text](https://nlp.cs.vcu.edu/images/vcu_head_logo "VCU")
- [Nanoinformatics Vertically Integrated Projects](https://rampages.us/nanoinformatics/)
