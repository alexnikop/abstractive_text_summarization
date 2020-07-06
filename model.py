import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope

class Model(object):
    def __init__(self, embeddings, FLAGS, word_dict_len):
        self.embeddings = embeddings
        self.word_dict_len = word_dict_len
        self.FLAGS = FLAGS
        self.article_max_len = 50
        self.summary_max_len = 15
        self.encoder_state = None
        self.encoder_output = None
        self.decoder_output = None
        self.logits = None
        self.loss = None
        self.update = None
        self.prediction = None
        self.mul_val = 1
        
        self.trunc_norm_init = tf.truncated_normal_initializer(1e-4)
        self.cell = tf.nn.rnn_cell.BasicLSTMCell
        #self.cell = tf.contrib.cudnn_rnn.CudnnLSTM
        self.projection_layer = tf.layers.Dense(word_dict_len, use_bias=False)
        self.reduce_layer = tf.layers.Dense(self.FLAGS.hidden_dim)

        #variables and constants
        self.global_step = tf.Variable(0, trainable=False)

        self.art_len = tf.constant(self.article_max_len)
        self.sum_len = tf.constant(self.summary_max_len)

        #placeholders
        self.batch_size = tf.placeholder(tf.int32, (), name="batch_size")
        self.enc_input = tf.placeholder(tf.int32, [None, self.article_max_len])
        self.dec_input = tf.placeholder(tf.int32, [None, self.summary_max_len])
        self.enc_input_lens = tf.placeholder(tf.int32, [None])
        self.dec_input_lens = tf.placeholder(tf.int32, [None])
        self.dec_output_labels = tf.placeholder(tf.int32, [None, self.summary_max_len])

        #embeddings
        self.encoder_embeddings = tf.transpose(tf.nn.embedding_lookup(
            self.embeddings, self.enc_input), perm=[1, 0, 2])
        self.decoder_embeddings = tf.transpose(tf.nn.embedding_lookup(
            self.embeddings, self.dec_input), perm=[1, 0, 2])
        
    with tf.device('/device:GPU:0'):
      def add_encoder(self):
          with tf.variable_scope("encoder", reuse = tf.AUTO_REUSE):
            fw_cells = [self.cell(self.FLAGS.hidden_dim) for _ in range(self.FLAGS.num_layers)]
            bw_cells = [self.cell(self.FLAGS.hidden_dim) for _ in range(self.FLAGS.num_layers)]
            #fw_cells = [self.cell(self.FLAGS.num_layers, self.FLAGS.hidden_dim)]
            #bw_cells = [self.cell(self.FLAGS.num_layers, self.FLAGS.hidden_dim)]
            if self.FLAGS.mode == "train":
              fw_cells = [rnn.DropoutWrapper(cell) for cell in fw_cells]
              bw_cells = [rnn.DropoutWrapper(cell) for cell in bw_cells]
            encoder_outputs, enc_fw_st, enc_bw_st = tf.contrib.rnn.stack_bidirectional_dynamic_rnn\
                (fw_cells, bw_cells, self.encoder_embeddings, time_major=True,
                 sequence_length=self.enc_input_lens, dtype=tf.float32)

            self.encoder_output = tf.concat(encoder_outputs, 2)
            encoder_state_c = tf.concat((enc_fw_st[0].c, enc_bw_st[0].c), 1)
            encoder_state_h = tf.concat((enc_fw_st[0].h, enc_bw_st[0].h), 1)
            if self.mul_val == 2:
              self.encoder_state = rnn.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)
            else:
              encoder_state_c_reduced = self.reduce_layer(encoder_state_c)
              encoder_state_h_reduced = self.reduce_layer(encoder_state_h)
              self.encoder_state = rnn.LSTMStateTuple(c=encoder_state_c_reduced, h=encoder_state_h_reduced)
            #self.encoder_state = self.reduce_states(enc_fw_st, enc_bw_st)
          
      def add_decoder(self):
          with tf.variable_scope("decoder", reuse = tf.AUTO_REUSE) as decoder_scope: 
              dec_cell = self.cell(self.FLAGS.hidden_dim * self.mul_val)
              if self.FLAGS.mode == "train":
                  attention_states = tf.transpose(self.encoder_output, perm=[1, 0, 2])
                  attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                      self.FLAGS.hidden_dim * self.mul_val, attention_states, memory_sequence_length=self.enc_input_lens, normalize=True)
                  attention_cell = tf.contrib.seq2seq.AttentionWrapper(
                      dec_cell, attention_mechanism, attention_layer_size=self.FLAGS.hidden_dim * self.mul_val)

                  decoder_initial_state = attention_cell.zero_state(dtype=tf.float32, batch_size=self.batch_size)
                  decoder_initial_state = decoder_initial_state.clone(cell_state=self.encoder_state)
                  helper = tf.contrib.seq2seq.TrainingHelper(self.decoder_embeddings, self.dec_input_lens, time_major=True)
                  decoder = tf.contrib.seq2seq.BasicDecoder(attention_cell, helper, decoder_initial_state)
                  outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=True, scope=decoder_scope)

                  self.decoder_output = outputs.rnn_output
                  self.logits = tf.transpose(self.projection_layer(self.decoder_output), perm=[1, 0, 2])
                  self.logits = tf.concat([self.logits,
                                           tf.zeros([self.batch_size, self.summary_max_len - tf.shape(self.logits)[1],
                                                     self.word_dict_len])], axis=1)
              elif self.FLAGS.mode == "test":
                  tiled_encoder_output = tf.contrib.seq2seq.tile_batch(
                      tf.transpose(self.encoder_output, perm=[1, 0, 2]), multiplier=self.FLAGS.beam_width)

                  tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(self.encoder_state,
                                                                            multiplier=self.FLAGS.beam_width)
                  tiled_sequence_length = tf.contrib.seq2seq.tile_batch(self.enc_input_lens, multiplier=self.FLAGS.beam_width)
                  attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                      self.FLAGS.hidden_dim * self.mul_val, tiled_encoder_output, memory_sequence_length=tiled_sequence_length,
                      normalize=True)
                  attention_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell, attention_mechanism,
                                                                       attention_layer_size=self.FLAGS.hidden_dim * self.mul_val)
                  decoder_initial_state = attention_cell.zero_state(dtype=tf.float32,
                                                                    batch_size=self.batch_size * self.FLAGS.beam_width)
                  decoder_initial_state = decoder_initial_state.clone(cell_state=tiled_encoder_final_state)
                  decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=attention_cell, embedding=self.embeddings,
                                                                 start_tokens=tf.fill([self.batch_size], tf.constant(2)),
                                                                 end_token=tf.constant(3),
                                                                 initial_state=decoder_initial_state,
                                                                 beam_width=self.FLAGS.beam_width,
                                                                 output_layer=self.projection_layer)
                  outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                          decoder, output_time_major=True, maximum_iterations=self.summary_max_len, scope=decoder_scope)
                  self.prediction = tf.transpose(outputs.predicted_ids, perm=[1, 2, 0])
          
    def calc_loss_and_update(self):
        with tf.variable_scope("loss_and_update", reuse = tf.AUTO_REUSE):
          ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
              logits=self.logits, labels=self.dec_output_labels)
          with tf.device('/device:GPU:0'):
            weights = tf.sequence_mask(self.dec_input_lens, self.sum_len, dtype=tf.float32)
            self.loss = tf.reduce_sum(ce * weights / tf.to_float(self.batch_size))

            params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
            optimizer = tf.train.AdamOptimizer(self.FLAGS.learning_rate)
            #optimizer = tf.train.AdadeltaOptimizer(self.FLAGS.learning_rate)

            self.update = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

    def build(self):
        self.add_encoder()
        self.add_decoder()
        if self.FLAGS.mode == "train":
            self.calc_loss_and_update()
