from __future__ import print_function
import sys, os, subprocess
from char_rnn.char_rnn import CharLSTM
from char_rnn.process_and_train import index_corpus
from streetview.gsv_api_fn import random_providence_image
import tensorflow as tf


# *************************************************************************** #
if __name__ == '__main__':
    # *********************************************************************** #
    # Args

    if len(sys.argv) == 0:
        print('args: <model name> <image filename> [<title>]')
        sys.exit(0)

    model_name = sys.argv[1]
    image_fn = sys.argv[2]
    title = sys.argv[3] if len(sys.argv) == 4 else None

    # *********************************************************************** #
    # process data cuz we need the index dicts
    # TODO: future versions could cache these

    char_to_index, index_to_char, _ = index_corpus('./char_rnn/data/')

    # *********************************************************************** #
    # load an image.
    # if u have a title and an image already there, we'll use those.
    # else, we gonna get a rando

    if title is None or not os.path.isfile(image_fn):
        title = random_providence_image(image_fn)

    # *********************************************************************** #
    # Now, run im2txt and get the caption
    im_path = os.path.realpath(os.path.join('./', image_fn))
    caption = subprocess.check_output(
        'bazel-bin/im2txt/run_inference --input_files="%s"' % im_path,
        shell=True)
    # *********************************************************************** #
    # Poem seed

    seed = '%s %s ' % (title, caption)

    # *********************************************************************** #
    # Load up the old model and sample a poem

    with tf.Session() as sess:
        model = CharLSTM.load_from(sess, './char_rnn/saved_models/',
                                   name=model_name)
        poem = model.sample(sess, 500, seed, char_to_index, index_to_char)

    # *********************************************************************** #
    # Nice

    print(poem)
