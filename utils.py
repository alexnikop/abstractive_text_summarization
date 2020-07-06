import re
import random
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import numpy as np
import pickle
import os

def clean_str(sentence):
    sentence = re.sub("``", "\"", sentence.strip())
    sentence = re.sub("''", "\"", sentence)
    sentence = re.sub("[#,.-]*#", "NUMBER", sentence)
    sentence = re.sub("_", "", sentence)
    return sentence

def shuffle_datasets(*sets):
  data =list(zip(*sets))

  random.shuffle(data)
  return zip(*data)

def file_to_list(data_path):
    with open(data_path, "r") as f:
        lines = list()
        line = clean_str(f.readline())
        while line:
            lines.append(line)
            line = clean_str(f.readline()) 
        return lines
    
      
def get_sentences_from_files(articles_path, summaries_path, mode, split, vocab_feed=2000000, data_splits=4):
    article_list = file_to_list(articles_path)

    if mode == 'test':
      article_sentences = list(map(lambda d: word_tokenize(d), article_list))
      return article_sentences
    
    elif mode == 'train':

      summary_list = file_to_list(summaries_path)
      
      if split == 0:
        if not os.path.exists('./vocab_feed_tokens.pickle'):
          print('Tokenizing Vocabulary feed...')
          article_sentences = list(map(lambda d: word_tokenize(d), article_list[:vocab_feed]))
          summary_sentences = list(map(lambda d: word_tokenize(d), summary_list[:vocab_feed])) 
          with open('./vocab_feed_tokens.pickle', "wb") as f:
            pickle.dump((article_sentences, summary_sentences), f)
        else:
          with open('./vocab_feed_tokens.pickle', "rb") as f:
            article_sentences, summary_sentences = pickle.load(f)
        
      else:
        
        limit = len(summary_list) // data_splits
        
        article_sentences = list(map(lambda d: word_tokenize(d), article_list[((split-1)*limit) : (split*limit)]))
        summary_sentences = list(map(lambda d: word_tokenize(d), summary_list[((split-1)*limit) : (split*limit)]))        
        article_sentences, summary_sentences = shuffle_datasets(article_sentences, summary_sentences)

      return article_sentences, summary_sentences
   
  
def batch_iter(inputs, outputs, batch_size, num_epochs, batches_per_epoch):
    inputs = np.array(inputs)
    outputs = np.array(outputs)

    for epoch in range(num_epochs):
        for batch_num in range(batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(inputs))
            yield inputs[start_index:end_index], outputs[start_index:end_index]


def process_batches(x_batch, y_batch, word_dict, mode, summ_max_len=15):

    enc_input_lens = list(map(lambda x: len([y for y in x if y != 0]), x_batch))

    if mode == "train":

        dec_input = list(map(lambda x: [word_dict["<s>"]] + list(x), y_batch))
        dec_input_lens = list(map(lambda x: len([y for y in x if y != 0]), dec_input))

        dec_output_labels = list(map(lambda x: list(x) + [word_dict["</s>"]], y_batch))

        dec_input = list(
            map(lambda d: d + (summ_max_len - len(d)) * [word_dict["<padding>"]], dec_input))
        dec_output_labels = list(
            map(lambda d: d + (summ_max_len - len(d)) * [word_dict["<padding>"]], dec_output_labels))

        return x_batch, dec_input, enc_input_lens, dec_input_lens, dec_output_labels

    return x_batch, enc_input_lens
