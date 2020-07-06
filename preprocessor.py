from utils import *
from collections import Counter
from gensim.models import Word2Vec
from collections import defaultdict
from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec

class Preprocessor(object):

    def __init__(self, FLAGS, article_max_len=50, summary_max_len=15, vocab_feed=2000000, 
        vocab_size=50000, train_article_path = "train.article.txt", train_title_path = "train.title.txt", \
        test_article_path = "test.article.txt", test_title_path = "test.title.txt"):
        
        self.article_max_len = article_max_len
        self.summary_max_len = summary_max_len
        self.article_sentences = None
        self.summary_sentences = None
        self.word_dict = None
        self.embeddings = None
        self.FLAGS = FLAGS
        self.vocab_feed = vocab_feed
        self.vocab_size = vocab_size
        self.dict_and_embs = 'dict_and_embs'
        self.train_article_path = train_article_path
        self.train_title_path = train_title_path
        self.test_article_path = test_article_path
        self.test_title_path = test_title_path
      
    def build_dict(self):
        if self.FLAGS.mode == "train" and self.FLAGS.train_voc:
            word_list = list()
            article_sentences, summary_sentences = \
                get_sentences_from_files(self.train_article_path, self.train_title_path, self.FLAGS.mode, 0)
            
            article_feed = article_sentences[:self.vocab_feed]
            summary_feed = summary_sentences[:self.vocab_feed]
            
            for sentence in article_feed + summary_feed:
                for word in sentence:
                    word_list.append(word)
            word_freq_list = Counter(word_list).most_common(self.vocab_size)
            
            if not os.path.exists("./" + self.dict_and_embs):
              os.mkdir("./" + self.dict_and_embs)
  
            word_dict = dict()

            word_dict["<padding>"] = 0
            word_dict["unk"] = 1
            word_dict["<s>"] = 2
            word_dict["</s>"] = 3

            for word, _ in word_freq_list:
                if word != "unk":
                  word_dict[word] = len(word_dict)

            self.word_dict = word_dict
            with open("./" + self.dict_and_embs + "/word_dict.pickle", "wb") as f:
                pickle.dump(word_dict, f)

        else:

            with open("./content/" + self.dict_and_embs + "/word_dict.pickle", "rb") as f:
                word_dict = pickle.load(f)
            self.word_dict = word_dict

        return self.word_dict

    def get_init_embedding(self):
        if self.FLAGS.mode == "train" and self.FLAGS.train_voc:
            if not os.path.exists('./word_vectors.pickle'):
              glove_file = "glove.42B.300d.txt"
              word2vec_file = get_tmpfile("word2vec_format.vec")
              glove2word2vec(glove_file, word2vec_file)
              print("Loading Glove vectors...")
              word_vectors = KeyedVectors.load_word2vec_format(word2vec_file)
              with open("./word_vectors.pickle", "wb") as f:
                pickle.dump(word_vectors, f)
            else:
              with open("./word_vectors.pickle", "rb") as f:
                word_vectors = pickle.load(f)
                
            reversed_dict = {value: key for key, value in self.word_dict.items()}

            word_vec_list = list()
            for _, word in sorted(reversed_dict.items()):
                try:
                    word_vec = word_vectors.word_vec(word)
                except KeyError:
                    word_vec = np.zeros([self.FLAGS.embedding_size], dtype=np.float32)    
                    
                word_vec_list.append(word_vec)

            # Assign random vector to <s>, </s> token
            word_vec_list[2] = np.random.normal(0, 1, self.FLAGS.embedding_size)
            word_vec_list[3] = np.random.normal(0, 1, self.FLAGS.embedding_size)

            with open("./" + self.dict_and_embs + "/embeddings.pickle", "wb") as f:
                pickle.dump(word_vec_list, f)
        else:
            with open("./content/" + self.dict_and_embs + "/embeddings.pickle", "rb") as f:
                word_vec_list = pickle.load(f)

        return np.array(word_vec_list, dtype="float32")

    def get_train_test_datasets(self, split=0):
        
        if self.FLAGS.mode == "test":
            
            x = get_sentences_from_files(self.test_article_path, self.test_title_path, self.FLAGS.mode, split)
            
            unk_dict = defaultdict(list)            
            
            for sen_id in range(len(x)):
                sen = x[sen_id]
                for word_id in range(len(sen)):
                      temp = sen[word_id]
                      sen[word_id] = self.word_dict.get(sen[word_id], self.word_dict["unk"])
                      if sen[word_id] == 1:
                          unk_dict[sen_id].append(temp)
                x[sen_id] = sen

            x = list(map(lambda d: d[:self.article_max_len-1], x))
            x = list(map(lambda d: d + (self.article_max_len - len(d)) * [self.word_dict["<padding>"]], x))
            
            with open("unk_dict.pickle", "wb") as f:
                pickle.dump(unk_dict, f)
                
            return x, unk_dict

        elif self.FLAGS.mode == "train":
          
            x,y = get_sentences_from_files(self.train_article_path, self.train_title_path, self.FLAGS.mode, split)
            
            x = list(map(lambda d: list(map(lambda w: self.word_dict.get(w, self.word_dict["unk"]), d)), x))
            x = list(map(lambda d: d[:self.article_max_len-1], x))
            x = list(map(lambda d: d + (self.article_max_len - len(d)) * [self.word_dict["<padding>"]], x))

            y = list(map(lambda d: list(map(lambda w: self.word_dict.get(w, self.word_dict["unk"]), d)), y))
            y = list(map(lambda d: d[:(self.summary_max_len-1)], y))

            return x, y
