import sys
import os
import argparse
import codecs
import tensorflow as tf
from nmt.utils import evaluation_utils as e_utils

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--reference', type=str, default=os.path.join(os.getcwd(), 'nmt', 'data', 'newstest2014.tok.bpe.32000.de'),
                            help="Reference text to compare accuracy against.")

    parser.add_argument('--accuracy_log', type=str, default = 'mlperf_log_accuracy.json',
                            help="Accuracy log file")

    args = parser.parse_args()

    if not os.path.exists(args.reference):
        print("Could not find reference file {}. Please specify its location".format(args.reference))
        sys.exit(0)

    if not os.path.exists(args.accuracy_log):
        print("Could not find accuracy log file {}. Please specify its location".format(args.accuracy_log))
        sys.exit(0)



    ref = []

    with codecs.getreader("utf-8")(
        tf.gfile.GFile(args.reference, "rb")) as ifh:
        ref_sentences = ifh.readlines()
        ref = [e_utils._clean(s, "bpe").split(" ") for s in ref_sentences]

    with open(args.accuracy_log) as ifh:
        for line in ifh:
            # Simplistic way to extract records
            if not "{" in line:
                continue
            s_line = line.strip("\n").strip(",")
            record = eval(s_line)

            # Extract sentence and sentence ID
            sentence = (bytes.fromhex(record["data"])).decode("utf-8")
            trans = sentence.split(" ")
            sent_id = record["qsl_idx"]
            print("Reference: {}\nTranslation: {}".format(ref[sent_id], trans))
            print("--")