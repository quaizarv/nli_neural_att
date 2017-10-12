import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils
from seq2seq_nli_na import model_with_buckets, nli_embedding_attention_seq2seq

class NLI_NA_Seq2SeqModel(object):
  """Sequence-to-sequence model with attention and for multiple buckets.

  This class implements a multi-layer recurrent neural network as encoder,
  and an attention-based decoder. This is the same as the model described in
  this paper: http://arxiv.org/abs/1412.7449 - please look there for details,
  or into the seq2seq library for complete model implementation.
  This class also allows to use GRU cells in addition to LSTM cells, and
  sampled softmax to handle large output vocabulary size. A single-layer
  version of this model, but with bi-directional encoder, was presented in
    http://arxiv.org/abs/1409.0473
  and sampled softmax is described in Section 3 of the following paper.
    http://arxiv.org/abs/1412.2007
  """

  def __init__(self, oov_words_count, word_vectors, num_labels, buckets,
               size, num_layers, max_gradient_norm, batch_size,
               learning_rate=1E-4, dropout=False, l2_reg_strength=False,
               use_gru=False, forward_only=False):
    """Create the model.

    Args:
      train_words_count: Words whose embeddings need to be learnt
      word_vectors: word vectors. The first train_words_count rows have been
        randomly initialized (whose representations we have to yet learn). The
        rest are filled with vectors trained elsewhere (e.g. using word2vec on
        gnews corpus)
      buckets: a list of pairs (I, O), where I specifies maximum input length
        that will be processed in that bucket, and O specifies maximum output
        length. Training instances that have inputs longer than I or outputs
        longer than O will be pushed to the next bucket and padded accordingly.
        We assume that the list is sorted, e.g., [(2, 4), (8, 16)].
      size: number of units in each layer of the model.
      num_layers: number of layers in the model.
      max_gradient_norm: gradients will be clipped to maximally this norm.
      batch_size: the size of the batches used during training;
        the model construction is independent of batch_size, so it can be
        changed after initialization if this is convenient, e.g., for decoding.
      learning_rate: learning rate to start with.
      dropout: dropout probability for random drop of nuerons in the network
      l2_reg_strength: L2 regularization strength
      use_gru: if true, we use GRU cells instead of LSTM cells.
      forward_only: if set, we do not construct the backward pass in the model.
    """
    self.oov_words_count = oov_words_count
    self.buckets = buckets
    self.batch_size = batch_size

    # Create the internal multi-layer cell for our RNN.
    single_cell = tf.nn.rnn_cell.BasicLSTMCell(size)
    if use_gru:
        single_cell = tf.nn.rnn_cell.GRUCell(size)
    cell = single_cell
    if num_layers > 1:
      cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)

    # The seq2seq function: we use embedding for the input, and attention.
    def seq2seq_f(prem_inputs, hypo_inputs, oov_prem_words, oov_hypo_words):
      return nli_embedding_attention_seq2seq(
          prem_inputs, hypo_inputs, oov_prem_words, oov_hypo_words, cell,
          word_vectors, num_labels, dropout_prob=dropout)

    # Feeds for inputs.
    self.prem_inputs = []
    self.hypo_inputs = []
    self.targets = []
    self.target_weights = []
    self.oov_prem_words = []
    self.oov_hypo_words = []

    for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
      self.prem_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="prem{0}".format(i)))
      self.oov_prem_words.append(tf.placeholder(
        tf.float32, shape=[None],
        name="oov_prem_words{0}".format(i)))
      
    for i in xrange(buckets[-1][1] + 1):
      self.hypo_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="hypo{0}".format(i)))
      self.oov_hypo_words.append(tf.placeholder(
        tf.float32, shape=[None],
        name="oov_hypo_words{0}".format(i)))
      self.targets.append(tf.placeholder(tf.int32, shape=[None],
                                        name="target{0}".format(i)))
      self.target_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                name="weight{0}".format(i)))


    # Training outputs and losses.
    self.outputs, self.losses = model_with_buckets(
      self.prem_inputs, self.hypo_inputs, self.targets, self.target_weights,
      self.oov_prem_words, self.oov_hypo_words, buckets,
      lambda w, x, y, z: seq2seq_f(w, x, y, z), l2_reg_strength=l2_reg_strength)

    
    # Gradients and SGD update operation for training the model.
    params = tf.trainable_variables()
    if not forward_only:
      self.gradient_norms = []
      self.updates = []
      opt = tf.train.AdamOptimizer(learning_rate)

      for b in xrange(len(buckets)):
        gradients = tf.gradients(self.losses[b], params)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                         max_gradient_norm)
        self.gradient_norms.append(norm)
        self.updates.append(opt.apply_gradients(
            zip(clipped_gradients, params)))
        
    self.saver = tf.train.Saver(tf.global_variables())

  def step(self, session, prem_inputs, hypo_inputs, targets, target_wts,
           oov_prem_words, oov_hypo_words,
           bucket_id, forward_only, run_metadata=None):
    # Check if the sizes match.
    prem_size, hypo_size = self.buckets[bucket_id]
    if len(prem_inputs) != prem_size:
      raise ValueError("Premise length must be equal to the one in bucket,"
                       " %d != %d." % (len(prem_inputs), prem_size))
    if len(hypo_inputs) != hypo_size:
      raise ValueError("Hypothesis length must be equal to the one in bucket,"
                       " %d != %d." % (len(hypo_inputs), hypo_size))
    if len(target_wts) != hypo_size:
      raise ValueError("Weights length must be equal to the one in bucket,"
                       " %d != %d." % (len(target_wts), hypo_size))

    # Input feed: prem inputs, hypothesis inputs, target_weights, as provided.
    input_feed = {}
    for l in xrange(prem_size):
      input_feed[self.prem_inputs[l].name] = prem_inputs[l]
      input_feed[self.oov_prem_words[l].name] = oov_prem_words[l]
    for l in xrange(hypo_size):
      input_feed[self.hypo_inputs[l].name] = hypo_inputs[l]
      input_feed[self.targets[l].name] = targets[l]
      input_feed[self.target_weights[l].name] = target_wts[l]
      input_feed[self.oov_hypo_words[l].name] = oov_hypo_words[l]

    
    # Output feed: depends on whether we do a backward step or not.
    if not forward_only:
      output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                     self.gradient_norms[bucket_id],  # Gradient norm.
                     self.losses[bucket_id]]  # Loss for this batch.
    else:
      output_feed = [self.losses[bucket_id]]  # Loss for this batch.
      for l in xrange(hypo_size):  # Output logits.
        output_feed.append(self.outputs[bucket_id][l])

    if run_metadata == None:
      outputs = session.run(output_feed, input_feed)
    else:
      outputs = session.run(output_feed, input_feed,
                            options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                            run_metadata=run_metadata)
    if not forward_only:
      return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
    else:
      return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.

  def prepare_batch(self, data, bucket_id, batch_id):
    prem_size, hypo_size = self.buckets[bucket_id]
    prem_inputs, hypo_inputs, targets, target_weights = [], [], [], []
    oov_prem_words, oov_hypo_words = [], []
                            
    # Get a random batch of prem and hypo inputs from data,
    # pad them if needed, reverse prem inputs and add GO to hypo.
    batch_start = self.batch_size * batch_id
    for offset in xrange(self.batch_size):
      prem_input, hypo_input, label = data[bucket_id][batch_start + offset]

      # Prem inputs are padded and then reversed.
      prem_pad = [data_utils.PAD_ID] * (prem_size - len(prem_input))
      prem_inputs.append(prem_input + prem_pad)

      # Hypo inputs get an extra "GO" symbol, and are padded then.
      hypo_pad_size = hypo_size - len(hypo_input) - 1
      hypo_inputs.append([data_utils.GO_ID] + hypo_input +
                            [data_utils.PAD_ID] * hypo_pad_size)

      # There is actually only one label which is the output of the the LSTM
      # RNN after processing the last word in the hypothesis.  But since the
      # hypothesis size in words in variable, we put a label at each time step
      # of the RNN but give a weight of zero to the outputs at every time step
      # except for the one corresponding to the last word in the hypothesis
      targets.append([label]*hypo_size)
      
      # Target weights are all zero except for the target corresponding to the
      # last token in the hypothesis
      weights = [0]*hypo_size
      weights[len(hypo_input)] = 1
      target_weights.append(weights)

      # Hack to take care of the fact that we have to learn the vector
      # representation of words which are not in the vocabulary of our
      # pre-trained vectors
      oov_prem_words.append([1 if w_id < self.oov_words_count else 0
                             for w_id in prem_inputs[-1]])
      oov_hypo_words.append([1 if w_id < self.oov_words_count else 0
                             for w_id in hypo_inputs[-1]])
        
    # Now we create batch-major vectors from the data selected above.
    batch_prem_inputs, batch_hypo_inputs = [], []
    batch_targets, batch_weights = [], []
    batch_oov_prem_words, batch_oov_hypo_words = [], []
    
    # Batch premise inputs are just re-indexed prem_inputs.
    for length_idx in xrange(prem_size):
      batch_prem_inputs.append(
          np.array([prem_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

    # Batch hypothesis inputs are re-indexed hypo_inputs.
    for length_idx in xrange(hypo_size):
      batch_hypo_inputs.append(
          np.array([hypo_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

    # Batch targets are re-indexed targets.
    for length_idx in xrange(hypo_size):
      batch_targets.append(
          np.array([targets[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

    # Batch weights are re-indexed target weights.
    for length_idx in xrange(hypo_size):
      batch_weights.append(
          np.array([target_weights[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.float32))

    # Batch oov_prem_words are re-indexed oov_prem_words
    for length_idx in xrange(prem_size):
      batch_oov_prem_words.append(
          np.array([oov_prem_words[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.float32))
      
    # Batch oov_hypo_words are re-indexed oov_hypo_words
    for length_idx in xrange(hypo_size):
      batch_oov_hypo_words.append(
          np.array([oov_hypo_words[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.float32))

    return (batch_prem_inputs, batch_hypo_inputs, batch_targets, batch_weights,
            batch_oov_prem_words, batch_oov_hypo_words)
