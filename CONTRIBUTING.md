# Contributing to medaCy
MedaCy seeks to create a unified platform to streamline research efforts in medical text mining while also providing an 
interface to easily apply models to real world problems. Due to this, contributions to medaCy are often consequences 
and direct by-products of active research projects. However, if not for the contributions, bug fixes/reports, 
and suggestions of practioners - medaCy could not grow and thrive.

This contribution guide is designed to inform:

1. **Researchers** in how they can efficiently utilize medaCy to make their work more reachable by practioners.
2. **Practioners** in how they can tune medaCy's cutting-edge functionalities to their specific application.

## Table of contents
1. [Issues and Bug Reports](#issues-and-bug-reports)
2. [Development Set-up](#development-environment-setup)
3. [Running Unit Tests](#running-unit-tests)

## Issues And Bug Reports
Please do a search before posting an issue/bug report - your problem may already be solved! If your search comes up for 
not - congratulations, you may have something to contribute!

## Development Environment Setup
At it's most basic one can fork medaCy, clone down their fork, and use their favorite text editor to develop. 
However, some up-front set-up effort goes a long way towards streamlining the contribution process and keeping organized.
This section details a suggested set-up for efficient development, testing, and experimentation with medaCy utilizing 
[PyCharm](https://www.jetbrains.com/pycharm/).

**Assumptions of this section:**
-  You are working in a UNIX based operating system.
-  Part 2 assumes you have Pycharm Professional installed - Pycharm Professional is provided with the Jetbrains 
University License. (this isn't entirely necessary but the useful Remote Host feature is disabled on the Community Edition)

**Part 1: Development Installation**
1. If you are shaky with git - [this link](https://nvie.com/posts/a-successful-git-branching-model/) provides an 
excellent description of the branching model medaCy follows to organize contributions.
2. Fork medaCy and copy the clone link.
3. On your machine, insure you have Python 3 installed. Set-up a [virtual environment](https://docs.python.org/3/library/venv.html) 
and activate it.
4. Run the bash commands: `python --version` and `pip list`. Upgrade pip to the latest version as suggested. 
Your python version should be above 3.4 and your installed packages should be few in number - if both of these 
conditions do not hold return to *Step 3*.
5. In a directory separate from the one created by the virtual envirorment set-up command, clone down your fork of medaCy.
6. Whilst inside your cloned fork, insure you are in at-least the *development* branch or a branch of the *development* branch. 
This can be verified by running `git status` and branching can be done with `git checkout <branch-name>`
7. Run `pip install -e .` This will install medaCy in editable mode inside of your virtual environment and will take 
several minutes to install dependencies - medaCy stands on the shoulders of giants! Errors one is likely to encounter 
here include the installation of sci-py and numpy. Google search the errors as they are easily fixable via the installation 
of some extra dependencies. Likely, your python installation is missing C headers required by scipy.

**Part 2: Developing with PyCharm**
PyCharm can streamline development efforts - especially if you are developing locally and running medaCy on a remote 
machine for model building.

**Part 3: Logging**

MedaCy uses the [logging](https://docs.python.org/3/howto/logging.html#logging-basic-tutorial) module to allow users 
insight into how medaCy is handling their data. Insure you are logging critical steps in any functionality you implement 
at the appropriate logging levels to make it easy for users to debug.

## Running Unit Tests
All components of medaCy have associated unit tests. Please insure these all pass before submitting pull requests. 
When medaCy runs unit tests, it first automatically installs the [END dataset](https://github.com/NanoNLP/medaCy_dataset_end) 
then uses it to test various functionalities of the package. Some tests involve building a model over the dataset - these 
may take some time to complete.

After installing medaCy for development, make sure that `pytest` is installed. Then:

1) For quick testing of the whole framework, run: \
    `python setup.py test`.
1) For more fine-grained testing on individual files with colorful log output run: \
    `pytest -s tests/tools/test_data_manager.py -o log_cli=True --log-cli-level=INFO`.

    This will show log output during tests and allow you to adust logging level for the test file being run. 
    Read the pytest documentation for details.

Note that some of the unit tests require knowledge about the configuration of your machine, and that those tests will
be skipped if those configuration settings are not specified in the config.json file. These settings include
the location of a MetaMap binary file on your machine, which GPU core to use for certain tests, and the location
of a word embeddings file. It may be that your contributions will not affect functionality that depend on these features,
however, all pull requests will be tested against the full unit test suite.
