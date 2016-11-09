from __future__ import print_function, division
import tensorflow as tf
import numpy as np


class CharLSTM():

    def __init__(embed_size, lstm_size, vocab_size, batch_size):
        # Set up embeddings

        # Set up 2-layer LSTM

        # Use seq2seq.rnn_decoder instead of dynamic_rnn to run the cells

        # softmax for logits

        # seq2seq loss by example
`
        # define trainer

        pass


    def train(sess, batches):
        pass


    def sample(sess, n, seed, vocab_index, chars):
        pass

