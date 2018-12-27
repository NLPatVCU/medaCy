from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
from medacy import __version__, __authors__
import sys

packages = find_packages()

def readme():
    with open('README.md') as f:
        return f.read()

class PyTest(TestCommand):
    """
    Custom Test Configuration Class
    Read here for details: https://docs.pytest.org/en/latest/goodpractices.html
    """
    user_options = [("pytest-args=", "a", "Arguments to pass to pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = ""

    def run_tests(self):
        import shlex
        # import here, cause outside the eggs aren't loaded
        import pytest

        errno = pytest.main(shlex.split(self.pytest_args))
        sys.exit(errno)

setup(
    name='medacy',
    version=__version__,
    license='GNU GENERAL PUBLIC LICENSE',
    description='Medical Natural Language Processing (NLP) with spaCy',
    long_description=readme(),
    packages=packages,
    url='https://github.com/NanoNLP/medaCy',
    author=__authors__,
    author_email='contact@andriymulyar.com',
    keywords='natural-language-processing medical-natural-language-processing machine-learning nlp-library metamap clinical-text-processing',
    classifiers=[
        '( Status :: 4 - Beta',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Programming Language :: Python :: 3.5',
        'Natural Language :: English',
        'Topic :: Text Processing :: Linguistic',
        'Intended Audience :: Science/Research'
    ],
    dependency_links=[
        'https://github.com/NanoNLP/medaCy_dataset_end/archive/v1.0.2.tar.gz#egg=medacy_dataset_end-1.0.2',
        'https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.0.0/en_core_web_sm-2.0.0.tar.gz#egg=en_core_web_sm-2.0.0'
    ],
    install_requires=[
        'spacy==2.0.13',
        'scikit-learn>=0.20.0',
        'sklearn-crfsuite',
        'xmltodict>=0.11.0',
        'joblib>=0.12.5',
        'tabulate>=0.8.2',
        'pathos>=0.2.2.1',
        'msgpack>=0.3.0,<0.6',
        'en_core_web_sm'
    ],
    tests_require=["pytest", "medacy_dataset_end"],
    cmdclass={"pytest": PyTest},
    include_package_data=True,
    zip_safe=False

)
