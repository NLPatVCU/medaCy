[![spaCy](https://img.shields.io/badge/built%20with-spaCy-09a3d5.svg)](https://spacy.io)

# medaCy
:hospital: Medical Text Mining and Information Extraction with spaCy :hospital:

MedaCy is a text processing and learning framework built over [spaCy](https://spacy.io/) to support the lightning fast 
prototyping, training, and application of highly predictive medical NLP models. It is designed to streamline researcher 
workflow by providing utilities for model training, prediction and organization while insuring the replicability of systems.

![alt text](https://nlp.cs.vcu.edu/images/Edit_NanomedicineDatabase.png "Nanoinformatics")

# :star2: Features
- Highly predictive, shared-task dominating out-of-the-box trained models for medical named entity recognition.
- Customizable pipelines with detailed development instructions and documentation.
- Allows the designing of replicable NLP systems for reproducing results and encouraging the distribution of models whilst still allowing for privacy.
- Active community development spearheaded and maintained by [NLP@VCU](https://nlp.cs.vcu.edu/).
- Detailed [API](https://medacy.readthedocs.io/en/latest/).

## :thought_balloon: Where to ask questions

MedaCy is actively maintained by a team of researchers at Virginia Commonwealth University. The best way to
receive immediate responses to any questions is to raise an issue. Make sure to first consult the 
[API](https://medacy.readthedocs.io/en/latest/).  See how to formulate a good issue or feature request in the [Contribution Guide](CONTRIBUTING.md).

## :computer: Installation Instructions
MedaCy can be installed for general use or for pipeline development / research purposes.

| Application | Run           |
| ----------- |:-------------:|
| Prediction and Model Training (stable) | `pip install git+https://github.com/NLPatVCU/medaCy.git` |
| Prediction and Model Training (latest) | `pip install git+https://github.com/NLPatVCU/medaCy.git@development` |
| Pipeline Development and Contribution  | [See Contribution Instructions](/CONTRIBUTING.md) |


# :books: Power of medaCy
After installing medaCy and [medaCy's clinical model](guide/models/clinical_notes_model.md), simply run:

```python
from medacy.model.model import Model

model = Model.load_external('medacy_model_clinical_notes')
annotation = model.predict("The patient was prescribed 1 capsule of Advil for 5 days.")
print(annotation)
```
and receive instant predictions:
```python
[
    ('Drug', 40, 45, 'Advil'),
    ('Dosage', 27, 28, '1'), 
    ('Form', 29, 36, 'capsule'),
    ('Duration', 46, 56, 'for 5 days')
]
```

MedaCy can also be used through its command line interface, documented [here](./guide/command_line_interface.md)

To explore medaCy's other models or train your own, visit the [examples section](guide).

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
Current contributors: Steele Farnsworth, Anna Conte, Gabby Gurdin, Aidan Kierans, Aidan Myers, and Bridget T. McInnes

Former contributors: Andriy Mulyar, Jorge Vargas, Corey Sutphin, and Bobby Best

Acknowledgments
===============
- [VCU Natural Language Processing Lab](https://nlp.cs.vcu.edu/) ![alt text](https://nlp.cs.vcu.edu/images/vcu_head_logo "VCU")
- [Nanoinformatics Vertically Integrated Projects](https://rampages.us/nanoinformatics/)
