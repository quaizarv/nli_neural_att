import tensorflow as tf

linear = tf.nn.rnn_cell._linear  # pylint: disable=protected-access

def attention_decoder(decoder_inputs, initial_state, attention_states, cell,
                      output_size=None, dtype=tf.float32, scope=None,
                      initial_state_attention=False):
  if not decoder_inputs:
    raise ValueError("Must provide at least 1 input to attention decoder.")
  if not attention_states.get_shape()[1:2].is_fully_defined():
    raise ValueError("Shape[1] and [2] of attention_states must be known: %s"
                     % attention_states.get_shape())

  if output_size is None:
    output_size = cell.output_size
    
  with tf.variable_scope(scope or "attention_decoder"):
    batch_size = tf.shape(decoder_inputs[0])[0]  # Needed for reshaping.
    attn_length = attention_states.get_shape()[1].value
    attn_size = attention_states.get_shape()[2].value

    # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
    hidden = tf.reshape(
        attention_states, [-1, attn_length, 1, attn_size])
    attention_vec_size = attn_size  # Size of query vectors for attention.
    k = tf.get_variable("AttnW", [1, 1, attn_size, attention_vec_size])
    hidden_features = tf.nn.conv2d(hidden, k, [1, 1, 1, 1], "SAME")
    v = tf.get_variable("AttnV", [attention_vec_size])

    state = initial_state

    def attention(query, last_step_attn):
      """Put attention masks on hidden using hidden_features and query."""
      with tf.variable_scope("Attention"):
          y = linear([query] + [last_step_attn], attention_vec_size, True)
          y = tf.reshape(y, [-1, 1, 1, attention_vec_size])
          # Attention mask is a softmax of v^T * tanh(...).
          s = tf.reduce_sum(
              v * tf.tanh(hidden_features + y), [2, 3])
          a = tf.nn.softmax(s)
          # Now calculate the attention-weighted vector d.
          with tf.variable_scope("WeightedPremiseRep"):
            d = tf.reduce_sum(
              tf.reshape(a, [-1, attn_length, 1, 1]) * hidden,
              [1, 2]) + \
              tf.tanh(
                  linear(last_step_attn, attention_vec_size, True))
          d = tf.reshape(d, [-1, attn_size])
      return d

    outputs = []
    batch_attn_size = tf.pack([batch_size, attn_size])
    attn = tf.zeros(batch_attn_size, dtype=dtype)
    
    # Ensure the second shape of attention vectors is set.
    attn.set_shape([None, attn_size])
    
    if initial_state_attention:
      attn = attention(initial_state)
    for i, inp in enumerate(decoder_inputs):
      if i > 0:
        tf.get_variable_scope().reuse_variables()

      # Run the RNN.
      cell_output, state = cell(inp, state)

      # Run the attention mechanism.
      if i == 0 and initial_state_attention:
        with tf.variable_scope(tf.get_variable_scope(),
                                           reuse=True):
          attn = attention(cell_output, attn)
      else:
        attn = attention(cell_output, attn)

      # Concatenate the cell output with attention and project
      with tf.variable_scope("FinalSentencePairRep"):
        final_sp_rep = tf.tanh(
          linear([cell_output] + [attn], attention_vec_size, True))

      # Compute the logits
      with tf.variable_scope("OuputSoftmax"):        
        output = linear(final_sp_rep, output_size, True)
        
      outputs.append(output)

  return outputs, state


def nli_embedding_attention_seq2seq(encoder_inputs, decoder_inputs,
                                    oov_prem_words, oov_hypo_words, cell,
                                    word_vectors, num_labels, dtype=tf.float32,
                                    dropout=0.0, scope=None,
                                    initial_state_attention=False,
                                    forward_only=False):
  
  with tf.variable_scope(scope or "nli_embedding_attn_seq2seq"):

    # Embedding Lookup shared by the premise/hypothesis stages
    '''embedding = tf.get_variable(
        "embedding",
        initializer=word_vectors)'''

    embedding_pretrained = tf.get_variable(
        "embedding_pretrained",
        initializer=word_vectors, trainable=False)
    
    '''
    encoder_inputs = [
      tf.nn.embedding_lookup(embedding, inp) *
      tf.reshape(oov_prem_words[i], [-1, 1]) + 
      tf.nn.embedding_lookup(embedding_pretrained, inp) *
      (1 - tf.reshape(oov_prem_words[i], [-1, 1]))
        for i, inp in enumerate(encoder_inputs)]
    decoder_inputs = [
      tf.nn.embedding_lookup(embedding, inp) *
      tf.reshape(oov_hypo_words[i], [-1, 1]) +
      tf.nn.embedding_lookup(embedding_pretrained, inp) * \
      (1 - tf.reshape(oov_hypo_words[i], [-1, 1]))
        for i, inp in enumerate(decoder_inputs)]
    '''
    encoder_inputs = [tf.nn.embedding_lookup(embedding_pretrained, inp)
                      for i, inp in enumerate(encoder_inputs)]
    decoder_inputs = [tf.nn.embedding_lookup(embedding_pretrained, inp)
                      for i, inp in enumerate(decoder_inputs)]

    cell = tf.nn.rnn_cell.InputProjectionWrapper(cell, cell.output_size)
    if not forward_only:
      keep_prob = 1.0 - dropout
      cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=keep_prob,
                                           output_keep_prob=keep_prob)
    encoder_outputs, encoder_state = tf.nn.rnn(
        cell, encoder_inputs, dtype=dtype)

    # First calculate a concatenation of encoder outputs to put attention on.
    top_states = [tf.reshape(e, [-1, 1, cell.output_size])
                  for e in encoder_outputs]
    attention_states = tf.concat(1, top_states)

    # Decoder.
    return attention_decoder(
        decoder_inputs, encoder_state, attention_states, cell,
        output_size=num_labels,
        initial_state_attention=initial_state_attention)


def model_with_buckets(encoder_inputs, decoder_inputs, targets, weights,
                       oov_prem_words, oov_hypo_words,
                       buckets, seq2seq, softmax_loss_function=None,
                       l2_reg_strength=0.0, name=None, reuse_=False):
  if len(encoder_inputs) < buckets[-1][0]:
    raise ValueError("Length of encoder_inputs (%d) must be at least that of la"
                     "st bucket (%d)." % (len(encoder_inputs), buckets[-1][0]))
  if len(targets) < buckets[-1][1]:
    raise ValueError("Length of targets (%d) must be at least that of last"
                     "bucket (%d)." % (len(targets), buckets[-1][1]))
  if len(weights) < buckets[-1][1]:
    raise ValueError("Length of weights (%d) must be at least that of last"
                     "bucket (%d)." % (len(weights), buckets[-1][1]))
  if len(oov_prem_words) < buckets[-1][1]:
    raise ValueError("Length of weights (%d) must be at least that of last"
                     "bucket (%d)." % (len(weights), buckets[-1][1]))

  all_inputs = encoder_inputs + decoder_inputs + targets + weights
  all_inputs += oov_prem_words, oov_hypo_words,
  losses = []
  outputs = []
  with tf.name_scope(name, "model_with_buckets", all_inputs):
    for j, bucket in enumerate(buckets):
      with tf.variable_scope(tf.get_variable_scope(),
                             reuse=True if reuse_ or j > 0 else None):
        bucket_outputs, _ = seq2seq(encoder_inputs[:bucket[0]],
                                    decoder_inputs[:bucket[1]],
                                    oov_prem_words[:bucket[0]],
                                    oov_hypo_words[:bucket[1]])

        outputs.append(bucket_outputs)
        loss = tf.nn.seq2seq.sequence_loss(
          outputs[-1], targets[:bucket[1]], weights[:bucket[1]],
          softmax_loss_function=softmax_loss_function)
        if l2_reg_strength > 0.0:
          variables = tf.trainable_variables()
          loss = loss + l2_reg_strength * tf.add_n([tf.nn.l2_loss(v)
                                                    for v in variables])
        losses.append(loss)

  return outputs, losses
