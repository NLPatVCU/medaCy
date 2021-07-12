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

## :computer: Installation Instructions
MedaCy supports Python >= 3.6

| Application | Run           |
| ----------- |:-------------:|
| Prediction and Model Training (stable) | `pip install git+https://github.com/NLPatVCU/medaCy.git` |
| Prediction and Model Training (latest) | `pip install git+https://github.com/NLPatVCU/medaCy.git@development` |

# :books: How to use medaCy

MedaCy's components can be imported into other Python programs, but is designed primarily to be used via its command line interface.
Once medaCy is installed, one can read the instructions at any time with this command.

```bash
python -m medacy --help
```

More thorough documentation is provided [here](./guide/command_line_interface.md).

## :thought_balloon: Where to Ask Questions

MedaCy is actively maintained by a team of researchers at Virginia Commonwealth University. The best way to
receive immediate responses to any questions is to open an issue in this repository.

Reference
=========
```bibtex
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
Current contributors: Steele Farnsworth, Gabby Gurdin, Aidan Myers, and Bridget T. McInnes

Former contributors: Andriy Mulyar, Jorge Vargas, Corey Sutphin, Bobby Best, Anna Conte, and Aidan Kierans

Acknowledgments
===============
- [VCU Natural Language Processing Lab](https://nlp.cs.vcu.edu/) ![alt text](https://nlp.cs.vcu.edu/images/vcu_head_logo "VCU")
- [Nanoinformatics Vertically Integrated Projects](https://rampages.us/nanoinformatics/)
