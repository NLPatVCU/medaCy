[![spaCy](https://img.shields.io/badge/built%20with-spaCy-09a3d5.svg)](https://spacy.io)
# medaCy
:hospital: Medical Natural Language Processing with spaCy :hospital:

MedaCy is a text processing and learning framework built over spaCy to support the lightning fast prototyping, building, and application of highly predictive named entity recognition and relationship extraction systems in the medical domain.

![alt text](https://nlp.cs.vcu.edu/images/Edit_NanomedicineDatabase.png "Nanoinformatics")

Features
========
- Highly predictive out-of-the-box trained models for clinical named entity recognition and relationship extraction.
- Customizable feature extraction pipelines for custom model building.
- Integrated converters for common text annotation formats (Prodigy, BRAT, etc).
- Pre-compiled medical terminology and abbreviation lexicons.


User Guide
==========
Using medaCy is simple: all one needs is to select a pipeline and provide it with training data to learn from.

Training a Named Entity Recognition model for Clinical Text using medaCy:

```python
from medacy.pipelines import ClinicalPipeline
from medacy.tools import DataLoader
from medacy.pipeline_component import MetaMap
import joblib

from medacy.learn import Learner

#Some more powerful pipelines require an outside knowledge source such as MetaMap.
metamap = MetaMap(metamap_path="/home/share/programs/metamap/2016/public_mm/bin/metamap")

#Automatically organizes your training files.
train_loader = DataLoader("/directory/containing/your/training/data/")

#Pre-metamap our training data to speed up building models.
train_loader.metamap(metamap)

#Create pipeline and specify entities to learn.
pipeline = ClinicalPipeline(metamap, entities=['Strength'])

#create a Learner using our pipeline and data
learner = Learner(pipeline, loader)

#Build a model (defaults to Conditional Random Field)
model = learner.train()
joblib.dump(model,'/location/to/save/model')
```

Prediction utilizing medaCy:
```python
from medacy.pipelines import ClinicalPipeline
from medacy.tools import DataLoader
from medacy.pipeline_component import MetaMap
import joblib

from medacy.predict import Predictor

model = joblib.load('/location/containing/saved/model')

#Some more powerful pipelines require an outside knowledge source such as MetaMap.
metamap = MetaMap(metamap_path="/home/share/programs/metamap/2016/public_mm/bin/metamap")

data_loader = DataLoader("/directory/containing/your/text/to/label")

#Pre-metamap our data we wish to label to speed up prediction. Not necessary.
data_loader.metamap(metamap)

pipeline = ClinicalPipeline(metamap, entities=['Strength'])

#create a Learner using our pipeline and data
predictor = Predictor(pipeline, data_loader, model=model)

predictor.predict()

#prediction appear in a /predictions sub-directory of your data.
```

An example combined pipeline script:
```python
from medacy.learn import Learner
from medacy.predict import Predictor
from medacy.pipelines import ClinicalPipeline
from medacy.tools import DataLoader
from medacy.pipeline_components import MetaMap
import logging, sys

#See what medaCy is doing at any part of the learning or prediction process
logging.basicConfig(stream=sys.stdout,level=logging.INFO) #set level=logging.DEBUG for more information

train_loader = DataLoader("/training/directory")
test_loader = DataLoader("/evaluation/directory")
metamap = MetaMap(metamap_path="/home/share/programs/metamap/2016/public_mm/bin/metamap")

train_loader.metamap(metamap)
test_loader.metamap(metamap)

pipeline = ClinicalPipeline(metamap, entities=['Drug', 'Form', 'Route', 'ADE', 'Reason', 'Frequency', 'Duration', 'Dosage', 'Strength'])

learner = Learner(pipeline, train_loader)

model = learner.train()

learner.cross_validate() #perform 10 fold cross validation on predicted model, this takes time.

predictor = Predictor(pipeline, test_loader, model=model)

predictor.predict()

#prediction appear in a /predictions sub-directory of your data.
```

Note, the ClinicalPipeline requires spaCy's small model - install it with pip:
```python
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.0.0/en_core_web_sm-2.0.0.tar.gz
```


Set-up
======
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

Contribution
============
To contribute do the following:
1) Enter into a python3 virtual envirorment, once inside make sure to upgrade pip to the latest version.
2) Fork and clone this repository, enter into the cloned repo and run:
```python
pip install -e .
```
This will install medaCy in editable mode. Any changes you make to medaCy sources code will be reflected immediately when used.

3) Insure you are developing in the development branch or your own branch of the development branch.


License
=======
This package is licensed under the GNU General Public License


Authors
=======
Andriy Mulyar, Bobby Best, Steele Farnsworth, Yadunandan Pillai, Corey Sutphin, Bridget McInnes

Acknowledgments
===============
- [VCU Natural Language Processing Lab](https://nlp.cs.vcu.edu/)     ![alt text](https://nlp.cs.vcu.edu/images/vcu_head_logo "VCU")
- [Nanoinformatics Vertically Integrated Projects](https://rampages.us/nanoinformatics/)