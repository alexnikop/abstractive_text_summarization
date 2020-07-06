from utils import *
import tensorflow as tf
import time

def train_model(model, preprocessor, word_dict, FLAGS, prev_folder_path='', data_splits=4, model_name='model', folder_path=''):
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
      
        if FLAGS.con_train:
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(prev_folder_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.global_variables())
            
        with open("logfile_" + model_name, "w") as f:
          for epoch in range(FLAGS.num_epochs):
            print("\n---------------------------------------------\n", file = f)
            print("\n---------------------------------------------\n")
            batches_per_epoch=0
            for split in random.sample(range(1, data_splits+1), data_splits):
              x_train, y_train = preprocessor.get_train_test_datasets(split)
              
              batches_per_split = (len(x_train) - 1) // FLAGS.batch_size + 1
              batches_per_epoch += batches_per_split
              batches = batch_iter(x_train, y_train, FLAGS.batch_size, 1, batches_per_split)
              print("\nNumber of batches for split {}: {}".format(split, batches_per_split))
              print("\nNumber of batches for split {}: {}".format(split, batches_per_split), file=f)
              prev = time.time()
              prev_1000 = time.time()
              avg_loss_epoch = 0
              avg_loss_1000 = 0
              step_counter=0
              try:
                for x_batch, y_batch in batches:
                  feed_data = process_batches(x_batch, y_batch, word_dict, "train")

                  train_feed_dict = {
                        model.batch_size: len(feed_data[0]),
                        model.enc_input: feed_data[0],
                        model.dec_input: feed_data[1],
                        model.enc_input_lens: feed_data[2],
                        model.dec_input_lens: feed_data[3],
                        model.dec_output_labels: feed_data[4]
                  }

                  _, step, loss = sess.run([model.update, model.global_step, model.loss], feed_dict=train_feed_dict)
                  avg_loss_1000 += loss
                  avg_loss_epoch += loss
                  step_counter += 1
                  if step % 1000 == 0:
                    time_1000 = round(time.time()-prev_1000, 5)
                    avg_loss_value = round(avg_loss_1000 / step_counter ,5)
                    print("step:{}  avg_loss:{}  time:{}".format(step, avg_loss_value, time_1000))
                    print("step:{}  avg_loss:{}  time:{}".format(step, avg_loss_value, time_1000),file=f)
                    prev_1000 = time.time()

                    step_counter = 0
                    avg_loss_1000 = 0
                    
              except KeyboardInterrupt:
                #saver.save(sess, model_folder + "/" + model_name + ".ckpt", global_step=step)
                sys.exit()
              
            if step % (batches_per_epoch) == 0:
              saver.save(sess, folder_path + "/" + model_name + ".ckpt", global_step=step)

        curr = time.time() - prev
        print('Training took {} seconds'.format(curr))
    
