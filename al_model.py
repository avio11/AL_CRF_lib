import sklearn_crfsuite
from sklearn_crfsuite import metrics

class AL_MODEL():
  def __init__(self, **kwargs):

    # Sets sampling strategy for the active learning algorithm
    if 'sampling' in kwargs:
      self.sampling = kwargs.get('sampling')
    else:
      self.sampling = least_confidence

    # # Defines number of samples to be queried per iteration
    # if 'query_size' in kwargs:
    #   self.query_size = kwargs.get('query_size')
    # else:
    #   self.query_size = 10

    # Sets function to predict NER tags given sentence (Or sets of sentences)
    if 'predict' in kwargs:
      self.predict = kwargs.get('predict')
    else:
      self.predict = ''

    # # Set epochs for supervised training with labeled set
    # if 'max_iterations' in kwargs:
    #   self.max_iterations = kwargs.get('max_iterations')
    # else:
    #   self.max_iterations = 100

    # Sets verbose for the supervised training
    if 'verbose' in kwargs:
      self.verbose = kwargs.get('verbose')
    else:
      self.verbose = False

    # Defines the model for the predictions
    if 'model' in kwargs:
      self.model = kwargs.get('model')
    else:
      self.model = sklearn_crfsuite.CRF(
          algorithm = 'l2sgd',
          c2 = 1,
          max_iterations = 100,
          all_possible_transitions = True,
          verbose = self.verbose
      )

  def train(self, x, y):
    self.model.fit(x, y)

  def prediction(self, corpus, sentence):
    if isinstance(sentence[0], list):
      feats = corpus.get_features(sentence)
      predictions = self.model.predict(feats)
    else:
      feats = corpus.get_sentence_features(sentence)
      predictions = self.model.predict_single(feats)
    return predictions

  def active_learning(self, corpus, max_iterations, query_size):
    performance = []
    total_data = len(corpus.labeled_x) + len(corpus.unlabeled)
    labeled_hist = []
    unlabeled_hist = []
    for i in range(max_iterations):
      print("Iteration:", i)
      self.train(corpus.labeled_x, corpus.labeled_y)
      idx = self.sampling(self.model, corpus.unlabeled, query_size)
      if idx==-1:
        break
      else:
        corpus.switch_set(idx)
      labeled_hist.append(len(corpus.labeled_x)/total_data)

      if corpus.testset:
        performance.append(eval(self.model, corpus.test_x, corpus.test_y, model_labels(self.model)))
    return performance, labeled_hist


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------ Utility functions -----------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Utility functions for the active learning model class

# Function to evaluate the model on test set / Code taken from "https://github.com/TeamHG-Memex/sklearn-crfsuite/blob/master/docs/CoNLL2002.ipynb" (Turned into function)
def eval(model, x_test, y_test, labels):
  y_pred = model.predict(x_test)
  return metrics.flat_f1_score(y_test, y_pred,
                      average='weighted', labels=labels)

# Function to extract label set from training set / Code taken from "https://github.com/TeamHG-Memex/sklearn-crfsuite/blob/master/docs/CoNLL2002.ipynb" (Turned into function)
def model_labels(model):
  labels = list(model.classes_)
  labels.remove('O')
  return labels

# Input: model, unlabeled data and query size
# Output: list of index for samples with the least confidence predictions
def least_confidence(model, unlabeled_x, batch_sample_size):
  # List containing (index, confidence) pairs
  idx_conf = []
  if not unlabeled_x:
    return -1
  for i in range(len(unlabeled_x)):
    idx_conf.append([i, get_confidence(model, unlabeled_x[i])])
  idx_conf.sort(key=lambda x: x[1])
  return [idx_conf[i][0] for i in range(min(len(unlabeled_x), batch_sample_size))]

# Helper function for the least confidence function (returns the probability of the chosen sentence sent)
def get_confidence(model, sent):
  aux = model.tagger_.tag(sent)
  return model.tagger_.probability(aux)


# Input: model and unlabeled data
# Output: list of index for samples with normalized least confidence predictions
def normalized_least_confidence(self, unlabeled):
  return -1
