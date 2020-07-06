from utils import *
import tensorflow as tf

def test_model(model, preprocessor, word_dict,  FLAGS, model_name='model'):
    reverse_dict = {value: key for key, value in word_dict.items()}
    x_test, unk_dict = preprocessor.get_train_test_datasets()
    num_batches_per_epoch = (len(x_test) - 1) // FLAGS.batch_size + 1
    print("Number of batches:", num_batches_per_epoch)
    batches = batch_iter(x_test, x_test, FLAGS.batch_size, 1, num_batches_per_epoch)
    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state("./content/" + model_name+ "/")
    global_counter = 0
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        saver.restore(sess, ckpt.model_checkpoint_path)
        with open("results_150_2_3_50k", "w") as f:
            for x_batch, _ in batches:
                feed_data = process_batches(x_batch, x_batch, word_dict, "test")

                test_feed_dict = {
                    model.batch_size: len(feed_data[0]),
                    model.enc_input: feed_data[0],
                    model.enc_input_lens: feed_data[1],
                }
                prediction = sess.run(model.prediction, feed_dict=test_feed_dict)
                prediction_output = list(map(lambda x: [reverse_dict[y] for y in x], prediction[:, 0, :]))


                if global_counter % 6400 == 0:
                    perc = round((global_counter * 100) / (num_batches_per_epoch * 64),2)
                    print(str(perc) +"% complete")
                
                
                for line in prediction_output:
                    summary = list()

                    for word in line:    
                        #if word == "unk":
                          #if unk_dict[global_counter]:
                          #  word = unk_dict[global_counter].pop(0) 
                            
                        if word == "</s>":
                            break
                        #if word not in summary and word != "<" and word != ">":
                        summary.append(word)
                      
                    f.write(" ".join(summary))
                    f.write("\n")
                    global_counter += 1
                    
        print("100% complete")
        