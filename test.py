import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from corpus import CORPUS
from al_model import AL_MODEL
from random import sample
# import utils


path = 'data/'

# Create corpus object
data_rand = CORPUS(
  labeled_path = path+'labeled_set.txt',
  unlabeled_path = path+'unlabeled_set.txt',
  data_structure = {'word': 0, 'ner': 3},
  toyset = True,
  testset = True,
  test_path = path+'/eng_testb.txt'
)

data_lc = CORPUS(
  labeled_path = path+'labeled_set.txt',
  unlabeled_path = path+'unlabeled_set.txt',
  data_structure = {'word': 0, 'ner': 3},
  toyset = True,
  testset = True,
  test_path = path+'/eng_testb.txt'
)

# Define new sampling function
def random_sampling(model, unlabeled_x, batch_sample_size):
  if not unlabeled_x:
    return -1
  else:
    return sample(range(min(len(unlabeled_x), batch_sample_size)), min(len(unlabeled_x), batch_sample_size))

# Create model object with the new defined random sampling function
m_rand = AL_MODEL(sampling = random_sampling)
performance_rand, labeled_set_size_rand = m_rand.active_learning(data_rand, 100, 60)


# Create model object with default sampling function (Least confidence)
m_lc = AL_MODEL()
performance_lc, labeled_set_size_lc = m_lc.active_learning(data_lc, 100, 60)

# Plot performance during training
plt.plot(labeled_set_size_lc, performance_lc, 'r--')
plt.plot(labeled_set_size_rand, performance_rand, 'b--')
red_patch = mpatches.Patch(color='red', label='LC')
blue_patch = mpatches.Patch(color='blue', label='RAND')
plt.legend(handles=[red_patch, blue_patch])
plt.show()

# Predict simple tags for sentence
a = "John is a professor in England"
a = a.split()
print(a)
print(m_lc.prediction(data_lc, a))
