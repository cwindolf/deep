'''
Load the model
Sample a poem
'''
from __future__ import print_function
from char_rnn import CharLSTM
from process_and_train import index_corpus
import tensorflow as tf
import sys


MODEL_SAVE_DIR = './saved_models/'


# *************************************************************************** #
if __name__ == '__main__':
    # *********************************************************************** #
    # args

    if len(sys.argv) not in (2, 3):
        print('Expected an argument, the poem seed.')
        print('usage: python3 sample.py <seed string here>')
        sys.exit(0)

    seed = sys.argv[1]

    # *********************************************************************** #
    # process data cuz we need the index dicts
    # TODO: future versions could cache these

    char_to_index, index_to_char, _ = index_corpus()

    # *********************************************************************** #
    # load model and sample
    # TODO: future versions could take n (200 below) as an argument

    with tf.Session() as sess:
        # do the loading
        model = CharLSTM.load_from(sess, MODEL_SAVE_DIR)
        # do the sampling
        sample = model.sample(sess, 200, seed, char_to_index, index_to_char)

    print(sample)
