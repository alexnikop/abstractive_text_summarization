import os
import re
import sys
import random
import numpy as np
from collections import Counter
!pip install gensim
from gensim.models import Word2Vec
from collections import defaultdict
from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
import pickle
import time
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope

train_article_path = "/content/gdrive/My Drive/summ_datasets_trimmed/train_article_trimmed"
train_title_path = "/content/gdrive/My Drive/summ_datasets_trimmed/train_title_trimmed"
test_article_path = "/content/gdrive/My Drive/summ_datasets_trimmed/test_article_trimmed"
test_title_path = "/content/gdrive/My Drive/summ_datasets_trimmed/test_title_trimmed"

train_article_path = "/content/gdrive/My Drive/summ_datasets/train.article.txt"
train_title_path = "/content/gdrive/My Drive/summ_datasets/train.title.txt"
test_article_path = "/content/gdrive/My Drive/summ_datasets/test.article.txt"
test_title_path = "/content/gdrive/My Drive/summ_datasets/test.title.txt"

dict_and_embs = 'dict_and_embs'
model_name = 'model'
vocab_feed = 2000000
vocab_size = 50000
data_splits = 4
prev_folder_path = "./content/{}".format(model_name)
folder_path = "./" + model_name
if not os.path.exists(folder_path):
  os.mkdir(folder_path)
  

article_max_len = 50
summary_max_len = 15

