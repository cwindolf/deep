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
        print('Expected an argument, the poem seed, and optional second')
        print('argument, the model name.')
        print('usage: python3 sample.py <seed string here> [<model name>]')
        sys.exit(0)

    seed = sys.argv[1]
    name = None if len(sys.argv) == 2 else sys.argv[2]

    # *********************************************************************** #
    # process data cuz we need the index dicts

    char_to_index, index_to_char, vocab_size = index_corpus()

    # *********************************************************************** #
    # load model and sample

    with tf.Session() as sess:
        if name is not None:
            model = CharLSTM.load_from(sess, MODEL_SAVE_DIR, name=name)
        else:
            model = CharLSTM.load_from(sess, MODEL_SAVE_DIR)
        sample = model.sample(sess, 200, seed, char_to_index, index_to_char)

    print(sample)