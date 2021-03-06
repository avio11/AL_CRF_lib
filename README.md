# AL NER library based on the sklearn-crfsuite implementation of the CRF model

Developing a library to help the fast implementation of active learning based systems for named entity recognition (with conditional random fields model).

Dependencies (Verified to work):
- sklearn_crfsuite (>=0.3)
- sklearn (>=0.23.1)
- matplolib (>=3.2.2)

<h2>First version:</h2>

- Defines classes AL_MODEL() and CORPUS().

- Allows change in the sampling function

- allows the use of toysets, where the whole dataset is labeled but a part of it is considered unlabeled (for testing of different active learning strategies)

- Allows user to provide a test set ,and therefore computes the performance of the model throughout the active learning process.

<h2>Second version (upcoming) </h2>
- Creates the annotation tool, allowing the manual annotation of the dataset during the active learning process (upcoming)

- Creates the possibility of using query-by-committee

- Creates the possibility to save final model

<h2> Test library with the provided code and data </h2>
Execute the following steps

1) Download/clone this repository

2) Extract it

3) run "python test.py" (tested on linux ubuntu)

<h3> Starter code </h3>
The implementation of the CRF uses the sklearn-crfsuite library, and part of the code was taken from the sklearn-crfsuite's github: https://github.com/TeamHG-Memex/sklearn-crfsuite/blob/master/docs/CoNLL2002.ipynb
