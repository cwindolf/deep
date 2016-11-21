from __future__ import print_function, division
import tensorflow as tf
from tensorflow.python.ops.nn import rnn_cell, dynamic_rnn, softmax
import numpy as np
from tqdm import tqdm
from os import path
import pickle


# *************************************************************************** #
# TF helpers

def weight(name, shape):
    return tf.get_variable(name, shape, 
                           initializer=tf.truncated_normal_initializer(stddev=0.1))


# *************************************************************************** #
# The model

class CharLSTM():

    def __init__(self, embed_size, lstm_size, vocab_size, \
                 batch_size, seq_length, learn_rate, \
                 keep_prob=0.75, num_layers=2, name='char_lstm'):
        '''
        Initialize a character-level multilayer LSTM language model.
        Arguments:
            @embed_size: dimensions of embedding space
            @vocab_size: number of things in vocabulary (characters!)
            @batch_size: sequences per training batch
            @seq_length: length of sequences in each training batch
            @learn_rate: AdamOptimizer step size
            @keep_prob:  1 - dropout probability
            @num_layers: number of LSTM cells to stack
        '''
        # store params
        self.embed_size, self.lstm_size = embed_size, lstm_size
        self.vocab_size, self.seq_length = vocab_size, seq_length
        self.batch_size, self.learn_rate = batch_size, learn_rate
        self.kp, self.num_layers = keep_prob, num_layers
        self.name = name

        # Placeholders for input/output and dropout
        self.train_inputs  = tf.placeholder(tf.int32, shape=[batch_size, seq_length])
        self.train_targets = tf.placeholder(tf.int32, shape=[batch_size, seq_length])
        self.sample_inputs = tf.placeholder(tf.int32, shape=[1, 1])
        self.keep_prob     = tf.placeholder(tf.float32)

        # Set up embeddings
        E = weight('embedding', [vocab_size, embed_size])
        train_embeddings = tf.nn.embedding_lookup(E, self.train_inputs, name='train_embeddings')
        sample_embeddings = tf.nn.embedding_lookup(E, self.sample_inputs, name='sample_embeddings')

        # TODO: dropout.

        # Set up 2-layer LSTM
        # Use dynamic_rnn to run the cells
        with tf.variable_scope('lstm') as scope:
            single_cell = rnn_cell.BasicLSTMCell(lstm_size)
            self.cell = rnn_cell.MultiRNNCell([single_cell] * num_layers,
                                              state_is_tuple=True)
            self.train_init_state = self.cell.zero_state(batch_size, tf.float32)
            self.sample_init_state = self.cell.zero_state(1, tf.float32)

            train_outputs, self.train_state = dynamic_rnn(self.cell, train_embeddings,
                                                          initial_state=self.train_init_state)
            scope.reuse_variables()
            sample_outputs, self.sample_state = dynamic_rnn(self.cell, sample_embeddings,
                                                            initial_state=self.sample_init_state)

        reshaped_train_outputs  = tf.reshape(train_outputs,
                                            (batch_size * seq_length, lstm_size))
        reshaped_sample_outputs = tf.reshape(sample_outputs, (1, lstm_size))

        # final feedforward layer (model logits)
        ff_weights = weight('ff_weights', (lstm_size, vocab_size))
        ff_biases = weight('ff_biases', (vocab_size,))
        train_logits = tf.add(ff_biases,
                              tf.matmul(reshaped_train_outputs, ff_weights))
        sample_logits = tf.add(ff_biases,
                               tf.matmul(reshaped_sample_outputs, ff_weights))
        self.probs = softmax(sample_logits)

        # softmax and loss for training
        log_perps = tf.nn.seq2seq.sequence_loss_by_example(
                                            [train_logits],
                                            [tf.reshape(self.train_targets, [-1])],
                                            [tf.ones([batch_size * seq_length])])
        self.loss = tf.reduce_sum(log_perps) / batch_size

        # define trainer, saver, inits
        self.train_op = tf.train.AdamOptimizer(learn_rate).minimize(self.loss)
        self.init_op = tf.initialize_all_variables()
        self.saver = tf.train.Saver()


    def train(self, sess, batches, num_epochs):
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
            this_state = sess.run(self.train_init_state)
            for i, t in tqdm(batches):
                batch_p, this_state, _ = sess.run([self.loss, self.train_state, self.train_op],
                                                  feed_dict={ self.train_inputs: i,
                                                              self.train_targets: t,
                                                              self.keep_prob: self.kp })
                epoch_perplexity += batch_p
            print('    Epoch %d Perplexity: %0.2f' % (e, epoch_perplexity / self.seq_length))


    def sample(self, sess, n, seed, char_to_index, index_to_char):
        '''
        Sample likely sentences starting with @seed from the language model.
        Arguments:
            @sess:          active tf.Session
            @n:             length of seq to generate
            @seed:          primer for generated text
            @char_to_index: dict from characters -> index numbers
            @index_to_char: dict from index numbers -> characters
        Returns:
            a sampled string of length `n` + len(`seed`)
        '''
        # Make sure seed ends in a space.
        if not seed[-1] == ' ':
            seed += ' '
        # Prime the LSTM layers by running the seed through
        # (except for the space at the end)
        this_state = sess.run(self.sample_init_state)
        for char in seed[:-1]:
            # QUESTION: why are they feeding in 1 at a time?
            #           TF does not like this coming in like this. very bad.
            i = np.full((1, 1), char_to_index[char], dtype=np.int32)
            feed = { self.sample_inputs: i, self.sample_init_state: this_state, self.keep_prob: 1.0 }
            this_state = sess.run(self.sample_state, feed_dict=feed)
        # Now, do the actual sampling
        poem, current_char = seed, seed[-1]
        for _ in range(n):
            i = np.full((1, 1), char_to_index[current_char], dtype=np.int32)
            feed = { self.sample_inputs: i, self.sample_init_state: this_state, self.keep_prob: 1.0 }
            probs, this_state = sess.run([self.probs, self.sample_state], feed_dict=feed)
            # might need p = probs[0]
            if current_char == ' ':
                sample = np.random.choice(self.vocab_size, p=probs[0])
            else:
                sample = np.argmax(probs)

            prediction = index_to_char[sample]
            poem += prediction
            current_char = prediction
        return poem


    def save_to(self, sess, model_save_dir):
        '''
        Write out this model's parameters and tf.Variable values.
        Parameters are saved using pickle to model_save_dir/self.name.defs.
        Tensorflow saves a checkpoint to model_save_dir
        Arguments:
            @sess:           active tf.Session()
            @model_save_dir: where to write everything to
        '''
        # save params to remake the graph
        with open(path.join(model_save_dir, self.name + '.defs'), 'wb') as defs:
            pickle.dump({
                            'embed_size' : self.embed_size,
                            'lstm_size'  : self.lstm_size,
                            'vocab_size' : self.vocab_size,
                            'seq_length' : self.seq_length,
                            'batch_size' : self.batch_size,
                            'learn_rate' : self.learn_rate,
                            'keep_prob'  : self.kp,
                            'num_layers' : self.num_layers,
                            'name'       : self.name,
                        }, defs)

        # save variable states
        self.saver.save(sess, path.join(model_save_dir, self.name + '.ckpt'))


    def _load_variables(self, sess, model_save_dir):
        '''
        Helper for `load_from` below, should never need to be called
        from elsewhere. Just loads up variables from tensorflow checkpoint.
        Arguments:
            @sess:           an active tf.Session()
            @model_save_dir: the directory where the tf checkpoint lives
        '''
        ckpt = tf.train.get_checkpoint_state(model_save_dir)
        self.saver.restore(sess, ckpt.model_checkpoint_path)


    @classmethod
    def load_from(cls, sess, model_save_dir, name='char_lstm'):
        '''
        Class method to load up an old model. 
        e.g. model = CharLSTM.load_from(...)
        Arguments:
            @sess:           active tf.Session()
            @model_save_dir: place to load old model from
            @name:           name that model was instantiated with
        Returns:
            an instance of CharLSTM if everything was OK. otherwise, bug out.
        '''
        # load old model's params
        with open(path.join(model_save_dir, name + '.defs'), 'rb') as defs:
            params = pickle.load(defs)
        # instantiate model with params
        model = cls(**params)
        # load up the old variables
        model._load_variables(sess, model_save_dir)
        return model
