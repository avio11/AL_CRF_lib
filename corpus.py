class CORPUS():
  # Labeled dataset
  labeled_x = []
  labeled_y = []
  # Unlabeled dataset (has labels if toyset == True)
  unlabeled = []
  unlabeled_y = []
  # Test set (loaded if testset == True)
  test_x = []
  test_y = []
  def __init__(self, **kwargs):

    # Defines whether whole dataset is labeled (Test active learning algorithms)
    if 'toyset' in kwargs:
      self.toyset = kwargs.get('toyset')
    else:
      self.toyset = False

    if 'testset' in kwargs:
      self.testset = kwargs.get('testset')
    else:
      self.testset = False

    # Load file paths for labeled set, unlabeled set and test set (if provided)
    assert 'labeled_path' in kwargs, "Typerror: missing path for labeled set file"
    self.labeled_path = kwargs.get('labeled_path')
    assert 'unlabeled_path' in kwargs, "Typerror: missing path for unlabeled set file"
    self.unlabeled_path = kwargs.get('unlabeled_path')

    if 'test_path' in kwargs:
      self.test_path = kwargs.get('test_path')
    else:
      self.test_path = ''

    # defines function to extract features from a sentence
    if 'extract_sentence_features' in kwargs:
      self.get_sentence_features = kwargs.get('extract_sentence_features')
    else:
      self.get_sentence_features = extract_features

    # Load structure of the file data (As dictionary of columns)
    assert 'data_structure' in kwargs, "Typerror: missing structure of labeled file"
    self.data_struct = kwargs.get('data_structure')

    # Load data (TODO: Test load data with toyset==False)-----------------------------------------
    self.load_data()
    # self.load_data(False)
    # Extract features from input data
    self.labeled_x = self.get_features(self.labeled_x)
    self.unlabeled = self.get_features(self.unlabeled)
    if self.test_x:
      self.test_x = self.get_features(self.test_x)

  # def load_data(self, is_labeled):
  #   if is_labeled:
  #     f = open(self.labeled_path, 'r').read()
  #     f = f.splitlines()
  #     self.labeled_x, self.labeled_y = split_sentences(f, self.data_struct)
  #   else:
  #     f = open(self.unlabeled_path, 'r').read()
  #     f = f.splitlines()
  #     if self.toyset:
  #       self.unlabeled, self.unlabeled_y = split_sentences(f, self.data_struct)
  #     else:
  #       self.unlabeled, _ = split_sentences(f, {})

  def load_data(self):
    # Load labeled dataset
    f = open(self.labeled_path, 'r').read()
    arq = f.splitlines()
    self.labeled_x, self.labeled_y = split_sentences(arq, self.data_struct)


    f = open(self.unlabeled_path, 'r').read()
    arq = f.splitlines()
    if self.toyset:
      self.unlabeled, self.unlabeled_y = split_sentences(arq, self.data_struct)
    else:
      self.unlabeled, _ = split_sentences(arq, {})


    if self.testset:
      f = open(self.test_path, 'r').read()
      arq = f.splitlines()
      self.test_x, self.test_y = split_sentences(arq, self.data_struct)

  def get_features(self, sentences):
    features = []
    for i in range(len(sentences)):
      sent = sentences[i]
      sent_feat = self.get_sentence_features(sent)
      features.append(sent_feat)
    return features

  def switch_set(self, idx):
    idx.sort(reverse=True)
    if self.toyset:
      for i in idx:
        self.labeled_x.append(self.unlabeled[i])
        self.labeled_y.append(self.unlabeled_y[i])
        del self.unlabeled[i]
        del self.unlabeled_y[i]
    else:
      for i in idx:
        self.labeled_x.append(self.unlabeled[i])
        self.labeled_y.append(self.request_label(i))
        del self.unlabeled[i]

  def request_label(self, idx):# TODO ------------------------------------------------------------------------------
    sent = self.unlabeled[idx]
    print("Sentence: ", sent)
    # TODO

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------ Utility functions -----------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Utility functions for the corpus class
def split_sentences(doc, data_struct):
  x = []
  y = []
  temp_x = []
  temp_y = []
  if not data_struct:
    for i in doc:
      if not i:
        x.append(temp_x)
        temp_x = []
      else:
        temp_x.append(i)
  else:
    for i in doc:
      if not i:
        x.append(temp_x)
        y.append(temp_y)
        temp_x = []
        temp_y = []
      else:
        i = i.split()
        temp_x.append(i[data_struct['word']])
        temp_y.append(i[data_struct['ner']])
  return x, y

# input: list containing one sentence split into words (e.g. ["EU", "is", "an", "organization"])
# output: List of dictionaries, each dictionary contains features for the words
def extract_features(sentence):
  sentence_features = []
  for j in range(len(sentence)):
    word_feat = {
            'word': sentence[j].lower(),
            'capital_letter': sentence[j].isupper(),
            'isdigit': sentence[j].isdigit(),
            'word_before': sentence[j].lower() if j==0 else sentence[j-1].lower(),
            'word_after:': sentence[j].lower() if j+1>=len(sentence) else sentence[j+1].lower(),
            'BOS': j==0,
            'EOS': j==len(sentence)-1
    }
    sentence_features.append(word_feat)
  return sentence_features
