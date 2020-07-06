import tensorflow as tf
from preprocessor import Preprocessor
from model import Model
from train_model import train_model
from test_model import test_model

FLAGS = tf.app.flags.FLAGS

# FLAGS
tf.app.flags.DEFINE_integer('hidden_dim', 256, 'dimension of RNN hidden states')
tf.app.flags.DEFINE_integer('num_layers', 1, 'depth of network')
tf.app.flags.DEFINE_integer('embedding_size', 300, 'size of word embeddings')
tf.app.flags.DEFINE_integer('beam_width', 10, 'beam size for beam search decoding')
tf.app.flags.DEFINE_float('learning_rate', 1e-3, 'learning_rate')
tf.app.flags.DEFINE_integer('batch_size', 64, 'size of each batch during training time')
tf.app.flags.DEFINE_integer('num_epochs', 3, 'number of epoch iterations')
tf.app.flags.DEFINE_boolean('train_voc', True, 'if True during training time, new Vocabulary and Word embeddings are calculated')
tf.app.flags.DEFINE_string('mode', "train", 'must be train or test')
tf.app.flags.DEFINE_boolean('con_train', True, 'if True, training continues from latest checkpoint')

def main(args):
    
    preprocessor = Preprocessor(FLAGS)
    
    word_dict = preprocessor.build_dict()
    embeddings = preprocessor.get_init_embedding()
    
    model = Model(embeddings, FLAGS, len(word_dict))
    model.build()
    
    if FLAGS.mode == "train":
        train_model(model, preprocessor, word_dict, FLAGS)
    elif FLAGS.mode == "test":
        test_model(model, preprocessor, word_dict, FLAGS)

    
if __name__ == "__main__":
    tf.app.run()

