from __future__ import print_function
from char_rnn import CharLSTM
import tensorflow as tf
import glob
import os


# *************************************************************************** #
# Constants

DATA_DIR = './data/'

# Model training params
NUM_EPOCHS = 20
EMBED_SIZE = 200
LSTM_SIZE  = 200
BATCH_SIZE = 20
SEQ_LENGTH = 20

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
            yield from line


def all_chars():
    '''
    Loop through all files in the whole `DATA_DIR`, yielding character
    by character.
    '''
    for filename in glob.iglob(os.path.join(DATA_DIR, '*.txt')):
        yield from characters(filename)



def index_corpus():
    '''
    Index all characters into integers, and also produce the reverse
    mapping and total number of unique characters.
    Returns:
        @n:             number of unique characters in corpus
        @char_to_index: dict mapping characters to their index numbers
        @index_to_char: dict with reverse mapping
    '''
    n, char_to_index, index_to_char = 0, {}, {}
    for char in all_chars():
        if char not in char_to_index:
            char_to_index[char] = n
            index_to_char[n] = char
            n += 1
    return char_to_index, index_to_char, n


def batch_windows(char_to_index):
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
    x_gen, y_gen = all_chars(), all_chars()
    next(y_gen)
    for x_char, y_char in zip(x_gen, y_gen):
        x_window.append(indexer[x_char])
        y_window.append(indexer[y_char])
        steps += 1
        # done with this window, add it to batch and restart
        if steps >= NUM_STEPS:
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

    char_to_index, index_to_char, vocab_size = index_corpus()

    # *********************************************************************** #
    # Instantiate and train the model

    model = CharLSTM(EMBED_SIZE, LSTM_SIZE, vocab_size, BATCH_SIZE, SEQ_LENGTH)

    with tf.Session() as sess:
        model.train(sess, batch_windows(char_to_index), NUM_EPOCHS)
        sample = model.sample(sess, 200, 'Providence is ',
                              char_to_index, index_to_char)

    # *********************************************************************** #
    # Test sample...

    print(sample)
