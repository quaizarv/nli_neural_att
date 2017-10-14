import os
import re
import collections
import csv
import pickle
import gensim
import numpy as np
import time

# Special vocabulary symbols - we always put them at the start.
_UNK = b"_UNK"
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_START_VOCAB = [_UNK, _PAD, _GO, _EOS]

UNK_ID = 0
PAD_ID = 1
GO_ID = 2
EOS_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")

# Class labels that we wish to predict
LABEL_ID = {'contradiction': 0, 'entailment': 1, 'neutral': 2}


def timeit(orig_fn):
    def new_fn(*args, **kwargs):
        t = time.time()
        ret = orig_fn(*args, **kwargs)
        print time.time() - t
        return ret
    return new_fn


def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
    return [w.lower() for w in words if w]


def test_basic_tokenizer():
    token_list = basic_tokenizer("Hi! How are you,")
    for t1, t2 in zip(token_list, ['hi', '!', 'how', 'are', 'you', ',']):
        assert(t1 == t2)


def read_data(filename, tokenizer=None):
    def tokenize(sentence):
        return tokenizer(sentence) if tokenizer else basic_tokenizer(sentence)
    data = []
    with open(filename, 'rb') as f:
        reader = csv.reader(f, delimiter='\t')
        # skip header
        reader.next()
        for row in reader:
            if row[0] == '-':
                continue
            parsed_row = [tokenize(sentence)
                          for sentence in (row[5], row[6])] + [row[0]]
            data.append(parsed_row)
    return data


def has_digits(w):
    if set(w) & set('0123456789'):
        return True
    return False


@timeit
def data_vocabulary(data, normalize_digits=True):
    def normalize(s):
        return [re.sub(_DIGIT_RE, b"0", w)
                if normalize_digits and has_digits(w) else w
                for w in s]
    words = []
    for wlist in [s1 + s2 for s1, s2, _ in data]:
        words.extend(wlist)
    words = normalize(words)
    vocab = collections.Counter(words)
    vocab_list = sorted(vocab, key=vocab.get, reverse=True)
    return vocab_list


@timeit
def word_vectors(vec_file, train_vocab_list):
    gnews_file = vec_file
    model = gensim.models.KeyedVectors.load_word2vec_format(gnews_file,
                                                            binary=True)

    # Sort the google-news trained word2vec by frequency and select words
    # which are either in top 50K by frequency or are in training
    # vocabulary
    sl = sorted([(k, model.vocab[k].index, model.vocab[k].count)
                 for k in model.vocab.keys()],
                key=lambda (k, idx, cnt): idx)
    freq_words = set(zip(*sl[:500000])[0]) - set(train_vocab_list)
    selected_vocab = train_vocab_list + list(freq_words)
    w2v = {w: model[w] for w in selected_vocab
           if w in model.vocab}
    return w2v


def sentence_to_token_ids(words, word2idx, normalize_digits=True):
  if not normalize_digits:
      return [word2idx.get(w, UNK_ID) for w in words]
  # Normalize digits by 0 before looking words up in the vocabulary.
  return [word2idx.get(re.sub(_DIGIT_RE, b"0", w), UNK_ID) for w in words]


def data_to_token_ids(data, word2idx, normalize_digits=True):
    result = []
    for s1, s2, label in data:
        result.append([sentence_to_token_ids(s, word2idx, normalize_digits)
                       for s in (s1, s2)] + [LABEL_ID[label]])
    return result


def counts(data):
    counts = collections.Counter((len(s1), len(s2)) for s1, s2, _ in data)
    return sorted(counts.items(), key=lambda (a, v): v, reverse=True)


def max_sentence_sizes(counts):
    max_first = max(zip(*zip(*counts)[0])[0])
    max_second = max(zip(*zip(*counts)[0])[1])
    return (max_first, max_second)


def data_stats(data_dir):
    snli_data_dir = os.path.join(data_dir, 'snli_1.0')
    train_file = os.path.join(snli_data_dir, 'snli_1.0_train.txt')
    dev_file = os.path.join(snli_data_dir, 'snli_1.0_dev.txt')
    test_file = os.path.join(snli_data_dir, 'snli_1.0_test.txt')

    train_data = read_data(train_file)
    dev_data = read_data(dev_file)
    test_data = read_data(test_file)

    train_counts = counts(train_data)
    dev_counts = counts(train_data)
    test_counts = counts(test_data)

    print 'max sentence sizes in training set: ', \
          max_sentence_sizes(train_counts)
    print 'max sentence sizes in dev set:      ', \
          max_sentence_sizes(dev_counts)
    print 'max sentence sizes in training set: ', \
          max_sentence_sizes(test_counts)

    prev_buckets_cum_sz = 0
    buckets = [(10, 10), (15, 10), (20, 15), (25, 20),
               (35, 30), (45, 40), (60, 45), (85, 65)]
    for b in buckets:
        new_cum_sz = sum(cnt for (s1, s2), cnt in train_counts
                         if s1 <= b[0] and s2 <= b[1])
        bucket_sz = new_cum_sz - prev_buckets_cum_sz
        prev_buckets_cum_sz = new_cum_sz
        print 'Data instances in bucket ({0}, {1}): {2}'.format(
            b[0], b[1], bucket_sz)


@timeit
def prepare_nli_data(data_dir, vec_file, tokenizer=None):
    processed_data_file = os.path.join(data_dir, 'snli_processed_data.pkl')
    if os.path.exists(processed_data_file):
        with open(processed_data_file, 'rb') as pkl_file:
            processed_data_dict = pickle.load(pkl_file)
    else:
        # Read data
        snli_data_dir = os.path.join(data_dir, 'snli_1.0')
        train_file = os.path.join(snli_data_dir, 'snli_1.0_train.txt')
        dev_file = os.path.join(snli_data_dir, 'snli_1.0_dev.txt')
        test_file = os.path.join(snli_data_dir, 'snli_1.0_test.txt')
        train_raw_data = read_data(train_file)
        dev_raw_data = read_data(dev_file)
        test_raw_data = read_data(test_file)

        # Collect Training data Vocabulary and extract corresponding word
        # vectors
        train_vocab_list = data_vocabulary(train_raw_data)
        w2v = word_vectors(vec_file, train_vocab_list)

        # We also extract some of the most common words found in the
        # Google News Word2Vec Model
        w2v_vocab = set(w2v.keys())
        vocab = _START_VOCAB + list(set(train_vocab_list) - w2v_vocab)
        train_words_count = len(vocab)
        vocab += list(w2v_vocab)
        word2idx = {w: i for i, w in enumerate(vocab)}

        word_vec_size = len(w2v[w2v.keys()[0]])
        i2v = np.zeros((len(word2idx), word_vec_size))
        for w, i in word2idx.items():
            if w in w2v:
                i2v[i] = w2v[w]
            else:
                i2v[i] = np.random.uniform(-0.5, 0.5, word_vec_size)

        train_data = data_to_token_ids(train_raw_data, word2idx)
        dev_data = data_to_token_ids(dev_raw_data, word2idx)
        test_data = data_to_token_ids(test_raw_data, word2idx)

        processed_data_dict = {}
        processed_data_dict['train_words_count'] = train_words_count
        processed_data_dict['train_vocab_list'] = train_vocab_list
        processed_data_dict['word2idx'] = word2idx
        processed_data_dict['i2v'] = i2v
        processed_data_dict['train_data'] = train_data
        processed_data_dict['dev_data'] = dev_data
        processed_data_dict['test_data'] = test_data

        with open(processed_data_file, 'wb') as pkl_file:
            pickle.dump(processed_data_dict, pkl_file)

    processed_data_items = ['train_words_count',
                            'train_vocab_list', 'word2idx', 'i2v',
                            'train_data', 'dev_data', 'test_data']
    return processed_data_dict


def test_prepare_nli_data(data_dir, vec_file):
    processed_data_file = os.path.join(data_dir, 'snli_processed_data.pkl')
    if os.path.exists(processed_data_file):
        os.remove(processed_data_file)
    pdd1 = prepare_nli_data(data_dir)
    assert os.path.exists(processed_data_file)
    pdd2 = prepare_nli_data(data_dir)

    assert pdd2['i2v'].shape[1] == 300
    assert pdd1['train_words_count'] == pdd1['train_words_count']

    # Check the consistency of word2idx mappings
    w2i1 = pdd1['word2idx']
    word2idx = w2i1
    w2i2 = pdd2['word2idx']
    assert sum(1 if w2i1[k] != w2i2[k] else 0 for k in w2i1.keys()) == 0

    # Check the consistency of train_vocab_lists
    tvl1 = pdd1['train_vocab_list']
    tvl2 = pdd2['train_vocab_list']
    assert sum(1 if w1 != w2 else 0
               for w1, w2 in zip(tvl1, tvl2)) == 0

    # Check that index to vector mappings are consistent
    i2v = pdd1['i2v']
    i2v2 = pdd2['i2v']
    assert sum(sum(1 if n1 != n2 else 0
                   for n1, n2 in zip(list(i2v2[k]), list(i2v[k])))
               for k in range(i2v.shape[0])) == 0

    PREMISE, HYPOTHESIS, LABEL = (0, 1, 2)
    # Check consistency of the train data
    td1 = pdd1['train_data']
    td2 = pdd2['train_data']
    for i, _ in enumerate(td1):
        sp1 = td1[i]
        sp2 = td2[i]
        for s_idx in [PREMISE, HYPOTHESIS]:
            assert sum(1 if w_id1 != w_id2 else 0
                       for w_id1, w_id2 in zip(sp1[s_idx], sp2[s_idx])) == 0
        assert sp1[LABEL] == sp2[LABEL]       # Label

    snli_data_dir = os.path.join(data_dir, 'snli_1.0')
    train_file = os.path.join(snli_data_dir, 'snli_1.0_train.txt')
    train_data_text_form = read_data(train_file)

    def word_to_id(w):
        return word2idx.get(re.sub(_DIGIT_RE, b"0", w), UNK_ID)
    LABEL_ID = {'contradiction': 0, 'entailment': 1, 'neutral': 2}
    for i, sp2 in enumerate(train_data_text_form):
        sp1 = td1[i]
        for s_idx in [PREMISE, HYPOTHESIS]:
            assert sum(1 if w_id != word_to_id(w) else 0
                       for w_id, w in zip(sp1[s_idx], sp2[s_idx])) == 0
        assert sp1[LABEL] == LABEL_ID[sp2[LABEL]]

    gnews_file = vec_file
    model = gensim.models.Word2Vec.load_word2vec_format(gnews_file,
                                                        binary=True)
    for k in word2idx.keys():
        if k in model:
            assert sum(i2v[word2idx[k]] == model[k]) == i2v.shape[1]


if __name__ == "__main__":
    data_dir = "/home/qv/nlp-data/SNLI/"
    vec_dir = "/home/qv/nlp-data/pretrained-vectors/"
    vec_file = os.path.join(vec_dir,
                            'GoogleNews-vectors-negative300.bin')

    test_prepare_nli_data(data_dir, vec_file)
