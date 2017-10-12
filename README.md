# Implementation of the "Reasoning about Entailment with Neural Attention" Paper
Authors: Tim Rocktaschel, Edward Grefenstette, Karl Moritz Hermann, Tomas Kocisky, Phil Blunsom

https://arxiv.org/pdf/1509.06664v1.pdf

## Description

Source code for natural language inference using the method described in the
above paper.

## Requirements
gensim
GoogleNews-vectors-negative300.bin
SNLI corpus: https://nlp.stanford.edu/projects/snli/

## Pretrained Word Vectors

Download pretrained word2vec vectors such as GoogleNews-vectors-negative300.bin
and set the location of corresponding embedding file in nli_neural_att.py.

> FLAGS.vec_dir = "./pretrained-vectors/"


The locations are specified by the "vec_dir"
parameters

## Set the location for the SNLI Corpus and training directory in the

Assuming that you have downloaded the SNLI corpus at ./SNLI and
the train directory is at ./train

> FLAGS.data_dir = "./SNLI/"

> FLAGS.train_dir = "./train/"


## Training
Set the mode paramter in nli_neural_att.py to 'train' and then run train_ldc.py
> FLAGS.mode = 'train'

> python nli_neural_att.py

Takes about 150 epochs (pretty fast on a GPU) to converge

# Testing
Set the mode paramter in nli_neural_att.py to 'test' and then run train_ldc.py
> FLAGS.mode = 'test'

> python nli_neural_att.py


