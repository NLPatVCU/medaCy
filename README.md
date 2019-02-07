[![spaCy](https://img.shields.io/badge/built%20with-spaCy-09a3d5.svg)](https://spacy.io)
# medaCy
:hospital: Medical Natural Language Processing with spaCy :hospital:

MedaCy is a text processing and learning framework built over [spaCy](https://spacy.io/) to support the lightning fast prototyping, training, and application of highly predictive medical NLP models. It is designed to streamline researcher workflow by providing utilities for model training, prediction and organization while insuring the replicability of systems.

![alt text](https://nlp.cs.vcu.edu/images/Edit_NanomedicineDatabase.png "Nanoinformatics")


# :star2: Features
- Highly predictive, shared-task dominating out-of-the-box trained models for medical named entity recognition.
- Customizable pipelines with detailed development instructions and documentation.
- Allows the designing of replicable NLP systems for reproducing results and encouraging the distribution of models whilst still allowing for privacy.
- Active community development spearheaded and maintained by [NLP@VCU](https://nlp.cs.vcu.edu/).
- Detailed [API](https://medacy.readthedocs.io/en/latest/)

## :thought_balloon: Where to ask questions

MedaCy is actively maintained by [@AndriyMulyar](https://github.com/AndriyMulyar)
and [@CoreySutphin](https://github.com/CoreySutphin). The best way to
receive immediate responses to any questions is to raise an issue. Make sure to first consult the [API](https://medacy.readthedocs.io/en/latest/).  See how to formulate a good issue or feature request in the [Contribution Guide](CONTRIBUTING.md).

## :computer: Installation Instructions
Medacy can be installed for general use or for pipeline development / research purposes.

First, make sure you have spaCy's small model installed: 

`pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.0.0/en_core_web_sm-2.0.0.tar.gz#egg=en_core_web_sm-2.0.0`

then

| Application | Run           |
| ----------- |:-------------:|
| Prediction and Model Training (stable) | `pip install git+https://github.com/NLPatVCU/medaCy.git` |
| Prediction and Model Training (latest) | `pip install git+https://github.com/NLPatVCU/medaCy.git@development` |
| Pipeline Development and Contribution  | [See Contribution Instructions](/CONTRIBUTING.md) |


Windows users should consider using Vagrant to run medaCy to ensure compatibility.
A guide can be found [here](./examples/guide/utilizing_vagrant.md). 

# :books: Power of medaCy
After installing medaCy and [medaCy's clinical model](examples/models/clinical_notes_model.md), simply run:

```python
from medacy.model import Model

model = Model.load_external('medacy_model_clinical_notes')
annotation = model.predict("The patient was prescribed 1 capsule of Advil for 5 days.")
print(annotation)
```
and receive instant predictions:
```python
{
    'entities': {
        'T3': ('Drug', 40, 45, 'Advil'),
        'T1': ('Dosage', 27, 28, '1'), 
        'T2': ('Form', 29, 36, 'capsule'),
        'T4': ('Duration', 46, 56, 'for 5 days')
     },
     'relations': []
}
```
To explore medaCy's other models or train your own, visit the [examples section](examples).

Reference
=========
```
@ARTICLE {
    author  = "Andriy Mulyar, Natassja Lewinski and Bridget McInnes",
    title   = "TAC SRIE 2018: Extracting Systematic Review Information with MedaCy",
    journal = "National Institute of Standards and Technology (NIST) 2018 Systematic Review Information Extraction (SRIE) > Text Analysis Conference",
    year    = "2018",
    month   = "nov"
}
```

License
=======
This package is licensed under the GNU General Public License.


Authors
=======
Andriy Mulyar, Corey Sutphin, Bobby Best, Steele Farnsworth, and Bridget T McInnes

Acknowledgments
===============
- [VCU Natural Language Processing Lab](https://nlp.cs.vcu.edu/)     ![alt text](https://nlp.cs.vcu.edu/images/vcu_head_logo "VCU")
- [Nanoinformatics Vertically Integrated Projects](https://rampages.us/nanoinformatics/)
