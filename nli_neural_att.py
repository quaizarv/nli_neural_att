import os
import sys
import time

import numpy as np
import tensorflow as tf

import data_utils
from model_nli_na import NLI_NA_Seq2SeqModel


class FLAGS():
  pass


FLAGS.mode = 'train'   # or 'test'
FLAGS.init_learning_rate = 1E-3
FLAGS.dropout = 0.2
FLAGS.l2_reg_strength = 0.0
FLAGS.batch_size = 64
FLAGS.size = 100
FLAGS.num_layers = 1
FLAGS.max_gradient_norm = 50.0
FLAGS.max_epochs
FLAGS.data_dir = "/home/qv/nlp-data/SNLI/"
FLAGS.train_dir = "/home/qv/nlp-data/nli-neural-att/"
FLAGS.vec_dir = "/home/qv/nlp-data/pretrained-vectors/"
FLAGS.interactive = False
FLAGS.vec_file = os.path.join(FLAGS.vec_dir,
                              'GoogleNews-vectors-negative300.bin')

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(11, 11), (16, 11), (21, 16), (85, 65)]


def bucketize_data(data):
  """Read data from source and target files and put into buckets.

  Args:
    source_path: path to the files with token-ids for the source language.
    target_path: path to the file with token-ids for the target language;
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).

  Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.
  """
  data_set = [[] for _ in _buckets]
  for premise, hypothesis, label in data:
    for bucket_id, (premise_size, hypothesis_size) in enumerate(_buckets):
      if len(premise) < premise_size and len(hypothesis) < hypothesis_size:
        data_set[bucket_id].append([premise, hypothesis, label])
        break
  return data_set


def create_model(session, forward_only, processed_data_dict):
  """Create translation model and initialize or load parameters in session."""

  i2v = processed_data_dict['i2v']
  word_vectors = np.array([i2v[i] for i in range(len(i2v))], dtype=np.float32)
  model = NLI_NA_Seq2SeqModel(
    processed_data_dict['train_words_count'],
    word_vectors,
    len(data_utils.LABEL_ID),
    _buckets,
    FLAGS.size, FLAGS.num_layers,
    FLAGS.max_gradient_norm, FLAGS.batch_size,
    FLAGS.init_learning_rate, FLAGS.dropout, FLAGS.l2_reg_strength,
    forward_only=forward_only)

  # Merge all the summaries and write them out to /tmp/train (by default)
  merged = tf.summary.merge_all()
  train_writer = tf.train.SummaryWriter(FLAGS.train_dir + '/train',
                                        graph=session.graph)
  #test_writer = tf.train.SummaryWriter(FLAGS.train_dir + '/test')

  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if ckpt and ckpt.model_checkpoint_path:
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.global_variables_initializer())

  return model


def shuffle_data_and_generate_tokens(data_set):
  # Shuffle the data instances in each bucket
  for bucket in data_set:
    np.random.shuffle(bucket)

  # Get the # of batches for each bucket and create a turn token for each
  # batch.  The token identifies the batch's bucket and batch # within the
  # bucket.
  tokens = []
  for bucket_id, bucket in enumerate(data_set):
    bucket_sz = len(bucket)
    num_batches = bucket_sz / FLAGS.batch_size
    for batch_id in range(num_batches):
      token = [bucket_id, batch_id]
      tokens.append(token)

  # Shuffle the tokens
  np.random.shuffle(tokens)
  return tokens


def run_batches(sess, model, data_set, forward_only, tokens):

  # Run the epoch by looping over the tokens
  loss = 0.0
  correct_predictions = 0
  for token in tokens:
    bucket_id, batch_id = token
    (prem_inputs, hypo_inputs, targets, wts,
     oov_prem_words, oov_hypo_words) = model.prepare_batch(data_set,
                                                           bucket_id, batch_id)
    _, step_loss, outputs = model.step(sess,
                                       prem_inputs, hypo_inputs, targets, wts,
                                       oov_prem_words, oov_hypo_words,
                                       bucket_id, forward_only)
    loss += step_loss

    if not forward_only:
      continue

    for word_idx, wts_list in enumerate(wts):
      for i, wt in enumerate(wts_list):
        if wt == 1:
          if np.argmax(outputs[word_idx][i]) == targets[word_idx][i]:
            correct_predictions += 1

  avg_loss = loss / len(tokens)
  instance_count = len(tokens) * FLAGS.batch_size
  accuracy = 1.0 * correct_predictions / instance_count
  return avg_loss, accuracy, correct_predictions, instance_count


def train():
  """Train a NLI neural-attention model using SNLI data."""
  # Prepare SNLI data.
  print("Preparing SNLI data in %s" % FLAGS.data_dir)
  processed_data_dict = data_utils.prepare_nli_data(FLAGS.data_dir,
                                                    FLAGS.vec_file)

  with tf.Session() as sess:
    # Create model.
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    model = create_model(sess, False, processed_data_dict)

    # Read data into buckets and compute their sizes.
    dev_set = bucketize_data(processed_data_dict['dev_data'])
    train_set = bucketize_data(processed_data_dict['train_data'])

    # This is the training loop.
    for epoch in range(50):

      print 'Epoch: ', epoch
      tokens = shuffle_data_and_generate_tokens(train_set)
      i = 0
      while True:
        start_time = time.time()
        loss, _, _, _ = run_batches(sess, model, train_set, False,
                                    tokens[i:i + 100])
        run_time = time.time() - start_time
        # Print statistics for the previous epoch.
        print("run-time %.2f cross-entropy loss %.2f" % (run_time, loss))
        i += 100
        dev_tokens = shuffle_data_and_generate_tokens(dev_set)
        loss, acc, cps, n = run_batches(sess, model, dev_set, True,
                                        dev_tokens[:10])
        print("  eval: loss %.2f, accuracy %.2f,"
              " correct predictions %d, instances %d" %
              (loss, acc, cps, n))

        if i >= len(tokens):
          break

      # Save checkpoint and zero timer and loss.
      checkpoint_path = os.path.join(FLAGS.train_dir, "nli_na.ckpt")
      model.saver.save(sess, checkpoint_path)

      # Print statistics for the previous epoch.
      # print "epoch-time %.2f cross-entropy loss %.2f" % (epoch_time, loss)
      dev_tokens = shuffle_data_and_generate_tokens(dev_set)
      loss, acc, cps, n = run_batches(sess, model, dev_set, True, dev_tokens)
      print("  eval: loss %.2f, accuracy %.2f, correct predictions %d,"
            " instances %d" %
            (loss, acc, cps, n))

      # Run evals on development set and print their perplexity.
      # eval_loss, accuracy = run_epoch(sess, model, dev_set, True)
      # print("  eval: loss %.2f, accuracy %.2f" % (eval_loss, accuracy))
      sys.stdout.flush()


def test():
  """Train a NLI neural-attention model using SNLI data."""
  # Prepare SNLI data.
  print("Preparing SNLI data in %s" % FLAGS.data_dir)
  processed_data_dict = data_utils.prepare_nli_data(FLAGS.data_dir,
                                                    FLAGS.vec_file)
  with tf.Session() as sess:
    # Create model.
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    model = create_model(sess, True, processed_data_dict)

    # Read data into buckets and compute their sizes.
    test_set = bucketize_data(processed_data_dict['test_data'])
    test_tokens = shuffle_data_and_generate_tokens(test_set)
    loss, acc, cps, n = run_batches(sess, model, test_set, True, test_tokens)
    print("  eval: loss %.2f, accuracy %.2f, correct predictions %d,"
          " instances %d" %
          (loss, acc, cps, n))

    sys.stdout.flush()


if __name__ == "__main__":
  initialize_parameters()
  if FLAGS.mode == 'test':
    test()
  else:
    train()
  
