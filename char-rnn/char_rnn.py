from __future__ import print_function, division
import tensorflow as tf
from tensorflow.python.ops.nn import rnn_cell, dynamic_rnn, softmax
import numpy as np


# *************************************************************************** #
# TF helpers

def weight(shape):
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1))


# *************************************************************************** #
# The model

class CharLSTM():

    def __init__(self, embed_size, lstm_size, vocab_size, \
                 batch_size, seq_length, num_layers=2):
        '''
        Initialize a character-level multilayer LSTM language model.
        Arguments:
            @embed_size: dimensions of embedding space
            @vocab_size: number of things in vocabulary (characters!)
            @batch_size: sequences per training batch
            @seq_length: length of sequences in each training batch
            @num_layers: number of LSTM cells to stack
        '''
        # Store parameters that we'll need later
        self.vocab_size = vocab_size
        self.seq_length = seq_length

        # Placeholders for input/output and dropout
        self.inputs    = tf.placeholder(tf.int32, shape=[batch_size, seq_length])
        self.targets   = tf.placeholder(tf.int32, shape=[batch_size, seq_length])
        self.keep_prob = tf.placeholder(tf.float32)

        # Set up embeddings
        E = weight([vocab_size, embed_size])
        embeddings = tf.nn.embedding_lookup(E, self.inputs)

        # Set up 2-layer LSTM
        cell = rnn_cell.BasicLSTMCell(lstm_size)
        self.cell = cell = rnn_cell.MultiRNNCell([cell] * num_layers, 
                                                 state_is_tuple=True)
        self.init_state = cell.zero_state(batch_size, tf.float32)

        # Use dynamic_rnn to run the cells
        outputs, self.state = dynamic_rnn(cell, embeddings,
                                          initial_state=self.init_state)
        reshaped_outputs = tf.reshape(outputs, 
                                     (batch_size * seq_length, lstm_size))

        # final feedforward layer (model logits)
        ff_weights = weight((lstm_size, vocab_size))
        ff_biases = weight((vocab_size,))
        logits = tf.add(ff_biases,
                        tf.matmul(reshaped_outputs, ff_biases))
        self.probs = softmax(logits)

        # softmax and loss
        log_perps = tf.nn.seq2seq.sequence_loss_by_example(
                                            [logits], 
                                            [tf.reshape(targets, [-1])],
                                            [tf.ones([batch_size * seq_length])])
        self.loss = tf.reduce_sum(log_perps) / batch_size

        # define trainer
        self.train_op = tf.train.AdamOptimizer(LEARN_RATE).minimize(loss)


    def train(sess, batches, num_epochs):
        '''
        Train the model. Prints perplexity once per epoch. May eventually
        save model checkpoints, if we get around to it.
        Arguments:
            @sess:        an active tf.Session()
            @batches:     an iterable of training windows. each one of the form:
                          [(input1, target1), (input2, target2), ...]
            @num_epochs:  number of times to run thru the `batches`
        '''
        # if we're going to run thru multiple times, make sure it's not a generator.
        if num_epochs > 1:
            batches = list(batches)
        # begin training
        print('Training!')
        for e in range(num_epochs):
            print('    Epoch %d:' % e)
            epoch_perplexity = 0.0
            this_state = sess.run(self.init_state)
            for i, t in batches:
                batch_p, this_state, _ = sess.run([self.loss, self.state, self.train_op], 
                                                  feed_dict={
                                                    self.inputs: i,
                                                    self.targets: t,
                                                    self.keep_prob: 0.5
                                                })
                epoch_perplexity += batch_p
            print('    Epoch %d Perplexity: %0.2f' % (e, epoch_perplexity / self.seq_length))


    def sample(sess, n, seed, char_to_index, index_to_char):
        '''
        Sample likely sentences starting with @seed from the language model.
        Arguments:
            @sess:          active tf.Session
            @n:             length of seq to generate
            @seed:          primer for generated text
            @char_to_index: dict from characters -> index numbers
            @index_to_char: dict from index numbers -> characters
        Returns:
            a sampled string of length `n`
        '''
        # Make sure seed ends in a space.
        if not seed[-1] == ' ':
            seed += ' '
        # Prime the LSTM layers by running the seed through
        # (except for the space at the end)
        this_state = sess.run(self.init_state)
        for char in prime[:-1]:
            # QUESTION: why are they feeding in 1 at a time?
            i = np.full((1, 1), vocab[char])
            feed = { self.inputs: i, self.init_state: this_state }
            this_state = sess.run(self.state, feed_dict=feed)
        # Now, do the actual sampling
        poem, current_char = seed, seed[-1]
        for _ in range(n):
            i = np.full((1, 1), vocab[current_char])
            feed = { self.inputs: i, self.init_state: this_state }
            probs, this_state = sess.run([self.probs, self.state], feed_dict=feed)
            # might need p = probs[0]
            if current_char == ' ':
                sample = np.random.choice(self.vocab_size, p=probs)
            else:
                sample = np.argmax(probs)

            prediction = chars[sample]
            poem += prediction
            current_char = prediction
        return poem
