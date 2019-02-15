# Wrapping data in a medaCy compatible package.

In essence, a medaCy Compatible dataset is a thin software layer that interfaces with an existing information extraction dataset. A template contains a directory structure along with a `setup.py` for integrating existing information extraction datasets into a distributable and maintable format. MedaCy compatible datasets manifest as python packages that can be installed with a package manager such as `pip`.

## 1. Clone down a copy of the MedaCy dataset template
MedaCy Dataset Git Repository: https://github.com/NLPatVCU/medaCy_dataset_template

**Renaming of files in the template will be done in Step 3**

The template is structured as follows:
```
├── LICENSE
├── MANIFEST.in
├── medacy_dataset_template
│   ├── data
│   │   ├── evaluation
│   │   │   └── README.md
│   │   └── training
│   │       └── README.md
│   ├── __init__.py
│   └── medacy_dataset_template.py
├── README.md
└── setup.py
```

- `LICENSE` contains a copy of the GPL 3 License
- `MANIFEST.in` contains a line instructing the python package manager to not ignore the raw data files located in the `/data/` directory when installing the package.
- The sub-directory `medacy_dataset_template` contains the source code of the software layer that allows medaCy to interface with the data.
- The sub-directory `medacy_dataset_template/data` contains the actual dataset (raw text and corresponding annotation files) you are wrapping. Place designated training data in the `medacy_dataset_template/data/training` directory and designated evaluation data in the `medacy_dataset_template/data/evaluation` directory. If no designated evaluation data exists, place all data in the training directory. **This data is expected to be in BRAT ANN format**.
- `README.md` is a github flavored markdown file that should be filled out. It provides a high level overview of the dataset including the entitiy types, instance counts, original authors, any relevant citations, etc.
- `setup.py` contains the code allowing setuputils to build the python package.


## 2. Set-up your datasets Git Repository
Create a git (on github) repository with the name `medacy_dataset_<name>` where `<name>` is a all lowercase placeholder for the name of the dataset or data you are wrapping. Set this to private if you would like to restrict access to your data to select individuals. Clone this down in a seperate directory from the template.

## 3. Customizing the template for you dataset.
**Copy over the files from the cloned template repository into your local version of your dataset repository**
To customize the template one simply needs to insert the dataset files into the appropriate folders, change the string `medaCy_dataset_template` into `medaCy_dataset_<name>` through the package and update the main `README.md` file with model specific information.

Once data is inserted into the appropriate folders (reference Step 2), renaming will have to be done in/to the following file paths or inside the following files:

- `/setup.py`
- `/MANIFEST.in`
- `/README.md`
- `/medacy_dataset_template`
- `/medacy_dataset_template/__init__.py`
- `/medacy_dataset_template/medacy_dataset_template.py`

## 4. Test your package
Attempt to install your package with `pip`. If any errors are found, you did not fully update the template in the `setup.py`. Once installed, attempt to utilize the dataset as demonstrated in the `Dataset guide`. Any errors are due to not fully updating the template with strings

## 5. Distribute
Push your package to github. Now anyone can install it and work with your dataset. If an update needs to be made to the dataset, simply change the raw data files and update the git repository. Once a workable version is pushed, `tag` the commit and make a release to cement the dataset at that version.
