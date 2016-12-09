'''
Process the poems
Train the model
Save the model
'''
from __future__ import print_function
from char_rnn.char_rnn import CharLSTM
import tensorflow as tf
from tqdm import tqdm, trange
import glob, os


# *************************************************************************** #
# Constants

DATA_DIR = './data/'
MODEL_SAVE_DIR = './saved_models/'

# Model training params
NUM_EPOCHS = 150
EMBED_SIZE = 64
LSTM_SIZE  = 256
BATCH_SIZE = 20
SEQ_LENGTH = 50
LEARN_RATE = 1e-4

# Continuing to train old model?
LOAD_FROM_SAVE = True

# *************************************************************************** #
# Data processing

def characters(filename):
    '''
    Produce a generator looping through all characters in a file.
    Arguments:
        @filename: the file to generate characters from
    '''
    with open(filename, 'r') as file:
        for line in file:
            for char in line:
                yield char
            yield '\n'


def all_chars(data_dir):
    '''
    Loop through all files in the whole `data_dir`, yielding character
    by character.
    Every call runs through poems in a random order.
    '''
    for filename in glob.iglob(os.path.join(data_dir, '*.txt')):
        for char in characters(filename):
            yield char


def index_corpus(data_dir):
    '''
    Index all characters into integers, and also produce the reverse
    mapping and total number of unique characters.
    Returns:
        @n:             number of unique characters in corpus
        @char_to_index: dict mapping characters to their index numbers
        @index_to_char: dict with reverse mapping
    '''
    n, char_to_index, index_to_char = 0, {}, {}
    for char in all_chars(data_dir):
        if char not in char_to_index:
            char_to_index[char] = n
            index_to_char[n] = char
            n += 1
    return char_to_index, index_to_char, n


def batch_windows(char_to_index, data_dir):
    '''
    Batch the data into windows.
    Arguments:
        @char_to_index: mapping from characters to their indices
    Returns:
        generator of pairs of input and target batches
    '''
    x_batch, x_window = [], []
    y_batch, y_window = [], []
    windows, steps = 0, 0
    x_gen, y_gen = all_chars(data_dir), all_chars(data_dir)
    next(y_gen)
    for x_char, y_char in zip(x_gen, y_gen):
        x_window.append(char_to_index[x_char])
        y_window.append(char_to_index[y_char])
        steps += 1
        # done with this window, add it to batch and restart
        if steps >= SEQ_LENGTH:
            x_batch.append(x_window)
            y_batch.append(y_window)
            windows += 1
            x_window, y_window = [], []
            steps = 0
        # done with this batch, yield and restart
        if windows >= BATCH_SIZE:
            yield x_batch, y_batch
            x_batch, y_batch = [], []
            windows = 0



# *************************************************************************** #
if __name__ == '__main__':
    # *********************************************************************** #
    # Process data

    char_to_index, index_to_char, vocab_size = index_corpus(DATA_DIR)

    # *********************************************************************** #
    # Instantiate and train the model

    if not LOAD_FROM_SAVE:
        model = CharLSTM(EMBED_SIZE, LSTM_SIZE, vocab_size,
                         BATCH_SIZE, SEQ_LENGTH, LEARN_RATE, name= "cases")

    with tf.Session() as sess:
        # init model
        if not LOAD_FROM_SAVE:
            sess.run(model.init_op)
        else:
            model = CharLSTM.load_from(sess, MODEL_SAVE_DIR)

        # randomly shuffle order of poems every batch
        for e in trange(NUM_EPOCHS, desc='Training'):
            try:
                perp = model.train(sess, tqdm(list(batch_windows(char_to_index, DATA_DIR)),
                                              desc='    Epoch %d' % e))
                tqdm.write('Ep: %d - Perp: %0.2f' % (e, perp))
                tqdm.write('Sample from this epoch:')
                tqdm.write('    %s' % model.sample(sess, 100, 'Providence is ',
                                                   char_to_index, index_to_char))
            except KeyboardInterrupt:
                break

        # write out model
        model.save_to(sess, MODEL_SAVE_DIR)
