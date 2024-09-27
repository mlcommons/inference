# Copyright 2018 The MLPerf Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

from generic_loadgen import *
import sys
from nmt.nmt import create_hparams, add_arguments, create_or_load_hparams
import argparse
from nmt.inference import get_model_creator, start_sess_and_load_model, load_data
import tensorflow as tf
import os
from nmt.utils import misc_utils as utils
from nmt.utils import nmt_utils
from nmt import model_helper
import codecs
import array

NANO_SEC = 1e9

##
# @brief Translation task that contains 1 sentence ID.
class TranslationTask:
    def __init__(self, query_id, sentence_id, output_file):
        self.query_id = [query_id]
        self.sentence_id = sentence_id
        self.output_file = output_file
        self.start = time.time()

##
# @brief Translation task that contains an array of sentence IDs
class BatchTranslationTask:
    def __init__(self, sentence_id_list, query_id_list):
        self.sentence_id_list = sentence_id_list
        self.query_id_list = query_id_list
        self.query_id = query_id_list   #FIXME generic_loadgen needs this


##
# @brief Wrapper class around the TensorFlow GNMT reference implementation
class GNMTWrapper:
    ##
    # @brief Constructor will build the graph and set some wrapper variables
    # @param ckpt_path: path to the GNMT checkpoint
    # @param hparams_path: path to the parameters used to configure GNMT graph
    # @param vocab_prefix: Path to vocabulary file (note: don't add .en or .de suffixes)
    # @param outdir: Output directory to optionally write translations to
    # @param batch_size: batch size to use when processing BatchTranslationTasks
    def __init__(self, ckpt_path=None, hparams_path=None, vocab_prefix=None, outdir=None, batch_size=32):
        # If no value is provided for the construtor arguments, set defaults here
        if ckpt_path is None:
            ckpt_path = os.path.join(os.getcwd(), 'ende_gnmt_model_4_layer',
                        'translate.ckpt')

        if hparams_path is None:
            hparams_path= os.path.join(os.getcwd(), 'nmt', 'standard_hparams',
                             'wmt16_gnmt_4_layer.json')

        if vocab_prefix is None:
            vocab_prefix = os.path.join(os.getcwd(), 'nmt', 'data', 'vocab.bpe.32000')


        flags = self.parse_options(ckpt_path, hparams_path, vocab_prefix, outdir, batch_size)

        self.setup(flags)

        self.count = 0
        self.infer_data = []  # This will be filled by load_sentences


    ##
    # @brief Parse GNMT-specific options before setting up
    def parse_options(self, ckpt_path, hparams_path, vocab_prefix, outdir, batch_size):
        FLAGS = None
        # TBD remove argument parsing, and just have it return all default values.
        nmt_parser = argparse.ArgumentParser()
        add_arguments(nmt_parser)
        FLAGS, unparsed = nmt_parser.parse_known_args()

        # Some of these flags are never used and are just set for consistency
        FLAGS.num_workers = 1
        FLAGS.iterations = 1
        FLAGS.infer_batch_size = batch_size
        FLAGS.num_inter_threads = 1
        FLAGS.num_intra_threads = 1
        FLAGS.run = "accuracy" # Needs to be set to accuracy to generate output

        # Pass in inference specific flags
        FLAGS.ckpt = ckpt_path
        FLAGS.src = 'en'
        FLAGS.tgt = 'de'
        FLAGS.hparams_path = hparams_path
        FLAGS.out_dir = outdir
        FLAGS.vocab_prefix = vocab_prefix
        
        return FLAGS

    ##
    # @brief Configure hparams and setup GNMT graph 
    # @pre Requires output from parse_options
    def setup(self, flags):
        # Model output directory
        out_dir = flags.out_dir
        if out_dir and not tf.gfile.Exists(out_dir):
          tf.gfile.MakeDirs(out_dir)

        # Load hparams.
        default_hparams = create_hparams(flags)
        loaded_hparams = False
        if flags.ckpt:  # Try to load hparams from the same directory as ckpt
          ckpt_dir = os.path.dirname(flags.ckpt)
          ckpt_hparams_file = os.path.join(ckpt_dir, "hparams")
          if tf.gfile.Exists(ckpt_hparams_file) or flags.hparams_path:
                # Note: for some reason this will create an empty "best_bleu" directory and copy vocab files
                hparams = create_or_load_hparams(ckpt_dir, default_hparams, flags.hparams_path, save_hparams=False)
                loaded_hparams = True
        
        assert loaded_hparams

        # GPU device
        config_proto = utils.get_config_proto(
            allow_soft_placement=True,
            num_intra_threads=hparams.num_intra_threads,
            num_inter_threads=hparams.num_inter_threads)
        utils.print_out(
            "# Devices visible to TensorFlow: %s" 
            % repr(tf.Session(config=config_proto).list_devices()))


        # Inference indices (inference_indices is broken, but without setting it to None we'll crash)
        hparams.inference_indices = None
        
        # Create the graph
        model_creator = get_model_creator(hparams)
        infer_model = model_helper.create_infer_model(model_creator, hparams, scope=None)
        sess, loaded_infer_model = start_sess_and_load_model(infer_model, flags.ckpt,
                                                       hparams)

        # Parameters needed by TF GNMT
        self.hparams = hparams

        self.infer_model = infer_model
        self.sess = sess
        self.loaded_infer_model = loaded_infer_model

    ##
    # @brief Run translation on a number of sentence id's
    # @param sentence_id_list: List of sentence numbers to translate
    # @return Translated sentences
    def translate(self, sentence_id_list):
        infer_mode = self.hparams.infer_mode

        # Set input data and batch size
        with self.infer_model.graph.as_default():
            self.sess.run(
                self.infer_model.iterator.initializer,
                feed_dict={
                    self.infer_model.src_placeholder: [self.infer_data[i] for i in sentence_id_list],
                    self.infer_model.batch_size_placeholder: min(self.hparams.infer_batch_size, len(sentence_id_list))
                })

        # Start the translation
        nmt_outputs, _ = self.loaded_infer_model.decode(self.sess)
        if infer_mode != "beam_search":
          nmt_outputs = np.expand_dims(nmt_outputs, 0)

        batch_size = nmt_outputs.shape[1]
        assert batch_size <= self.hparams.infer_batch_size

        # Whether beam search is being used or not, we only want 1 final translation
        assert self.hparams.num_translations_per_input == 1

        translation = []
        for decoded_id in range(batch_size):
            translation += [nmt_utils.get_translation(
                        nmt_outputs[0],
                       decoded_id,
                       tgt_eos=self.hparams.eos,
                       subword_option=self.hparams.subword_option)]

        # Keeping track of how many translations happened
        self.count += len(translation)

        return translation

    def resetCount(self):
        self.count = 0

    def getCount(self):
        return self.count

    def getBatchSize(self):
        return self.hparams.infer_batch_size

    def load_sentences(self, input_file):
        self.infer_data = load_data(input_file, self.hparams)

##
# @brief Basic class in which LoadGen can store queries that will be processed by GNMT
class GNMTRunner (Runner):
    ##
    # @brief Constructor 
    # @param model: GNMTWrapper object
    # @param input_file: path to the input text
    # @param verbose: provide some information on the progress
    def __init__(self, model, input_file=None, verbose=False):
        Runner.__init__(self)

        # If no value is provided for the construtor arguments, set defaults here
        if input_file is None:
            input_file = os.path.join(os.getcwd(), 'nmt', 'data', 'newstest2014.tok.bpe.32000.en')

        self.gnmt = model
        self.input_file = input_file
        
        self.VERBOSE = verbose

    ##
    # @brief Load sentences into the infer_data array and warmup the network
    def load_samples_to_ram(self, query_samples):
        self.gnmt.load_sentences(self.input_file)

        # Warmup
        warmup_ids = list(range(self.gnmt.getBatchSize()))
        self.gnmt.translate(warmup_ids)

        # Reset translation count
        self.gnmt.resetCount()

    ##
    # @brief Calculate total number of sentences in the input file
    # @note this can be accelerated by returning len(self.gnmt.infer_data)
    # however, this requires load_samples_to_ram be called prior
    def getTotalNumSentences(self):
        with open(self.input_file) as ifh:
            for line_no, l in enumerate(ifh):
                pass
            return line_no + 1

    ##
    # @brief Invoke GNMT to translate the query qitem
    # @pre Ensure load_samples_to_ram was called to fill self.infer_data
    def process(self, qitem):
        bs = self.gnmt.getBatchSize()
        num_samples = len(qitem.sentence_id_list)

        translation = []

        # Split the samples over batches
        for i in range(0, num_samples, bs):
            cur_sentid_list = [index for index in qitem.sentence_id_list[i:min(i+bs, num_samples)]] 
            translation += self.gnmt.translate(cur_sentid_list)

        if self.VERBOSE:
            print("Performed {} translations".format(self.gnmt.getCount()))
        
        return translation

    ##
    # @brief Create a batched task and add it to the queue
    def enqueue(self, query_samples):
        if self.VERBOSE:
            print("Received query")
        query_id_list = [sample.id for sample in query_samples]
        sentence_id_list = [sample.index for sample in query_samples] 
        task = BatchTranslationTask(sentence_id_list, query_id_list)
        self.tasks.put(task)

    ##
    # @brief Serialize the result and give it to mlperf_loadgen
    # @param query_ids is a list of query ids that generated the samples
    # @param results is a list of UTF-8 encoded strings 
    # @note Because of Python's Garbage Collection, we need to call QuerySamplesComplete before returning
    def post_process(self, query_ids, results):
        response = []
        # To prevent the garbage collector from removing serialized data before the call to QuerySamplesComplete
        # we need to keep track of serialized data here.
        gc_hack = []    
        for res, q_id in zip(results, query_ids):
            result_arr = array.array('B', res)
            gc_hack.append(result_arr)
            r_info = result_arr.buffer_info()
            response.append(mlperf_loadgen.QuerySampleResponse(q_id, r_info[0], r_info[1]))

        # Tell loadgen that we're ready with this query
        mlperf_loadgen.QuerySamplesComplete(response)

##
# @brief Subclass of GNMTRunner, specialized for batch size 1
class SingleStreamGNMTRunner (GNMTRunner):
    ##
    # @brief Constructor 
    # @param model: GNMTWrapper object
    # @param input_file: path to the input text
    # @param store_translation: whether output should be stored
    # @param verbose: provide some information on the progress
    # @param outdir: Output directory to optionally write translations to
    def __init__(self, model, input_file=None, store_translation=False, verbose=False, outdir=None):
        GNMTRunner.__init__(self, model, input_file, verbose=verbose)

        # SingleStreamGNMTRunner only handles batch sizes of 1
        assert self.gnmt.getBatchSize() == 1
        self.store_translation = store_translation
        self.out_dir = outdir

    ##
    # @brief Invoke GNMT to translate the input file
    # @pre Ensure load_samples_to_ram was called to fill self.infer_data
    def process(self, qitem):
        if self.store_translation or self.VERBOSE:
            assert len(qitem.query_id) == 1
            msg = "translate {} (QID {}): Sentence ID {}".format(self.gnmt.getCount(), qitem.query_id[0], qitem.sentence_id)
            if self.store_translation:
                msg += " --> " + qitem.output_file
            print (msg)
       
        sentence_id = qitem.sentence_id 

        translation = self.gnmt.translate([sentence_id])

        if self.store_translation:
            assert len(translation) == 1
            self.write_output(translation[0], qitem.output_file)

        return translation

    ##
    # @brief Write translation to file
    def write_output(self, translation, trans_file):
          with codecs.getwriter("utf-8")(
              tf.gfile.GFile(trans_file, mode="wb")) as trans_f:
            trans_f.write((translation + b"\n").decode("utf-8"))

    ##
    # @brief Create a task and add it to the queue
    def enqueue(self, query_samples):
        assert len(query_samples) == 1
        sample = query_samples[0]
        sentence_id = sample.index
        output_file = os.path.join(self.out_dir, "sentence_{}_de".format(sample.index))

        task = TranslationTask(sample.id, sentence_id, output_file)
        self.tasks.put(task)

##
# @brief subclass of GNMTRunner, specialized for grouping multiple querries together
class ServerGNMTRunner(GNMTRunner):
    ##
    # @brief Constructor 
    # @param model: GNMTWrapper object
    # @param input_file: path to the input text
    # @param verbose: provide some information on the progress
    def __init__(self, model, input_file=None, verbose=False):
            GNMTRunner.__init__(self, model, input_file, verbose)

    ##
    # @brief Override the default handle_tasks loop for smart batching
    # @detail Instead of processing one qitem at a time (which represents a single query), we can aggregate them here.
    def handle_tasks(self):
        while True:
            # Block until an item becomes available
            qitem = self.tasks.get(block=True)

            # Stop processing if parent signaled that we're done
            if qitem is None:
                break

            sentence_id_list = qitem.sentence_id_list
            query_id_list = qitem.query_id

            # Aggregate querries until there are more samples than the batch size,
            # or until we aggregated all current qurries
            try:
                # @note that by definition, Server queries should have no more than 1 element
                # Therefore we don't need to worry that batched_querries would be come larger than 
                # the batch size
                while len(sentence_id_list) < self.gnmt.getBatchSize():
                    qitem = self.tasks.get(block=False)
                    # Stop processing if parent signaled that we're done
                    if qitem is None:
                        return

                    assert len(qitem.sentence_id_list) == 1
                    sentence_id_list += qitem.sentence_id_list
                    query_id_list += qitem.query_id
            except queue.Empty as e:
                pass

            batched_qitem = BatchTranslationTask(sentence_id_list, query_id_list)

            if self.VERBOSE:
                print("Aggregated {} single-sample querries.".format(len(batched_qitem.sentence_id_list)))

            results = self.process(batched_qitem)
            response = []

            # Call post_process on all samples
            self.post_process(batched_qitem.query_id, results)

            self.tasks.task_done()


if __name__ == "__main__":
    SCENARIO_MAP = {
    "SingleStream": mlperf_loadgen.TestScenario.SingleStream,
    "MultiStream": mlperf_loadgen.TestScenario.MultiStream,
    "Server": mlperf_loadgen.TestScenario.Server,
    "Offline": mlperf_loadgen.TestScenario.Offline,
    }

    MODE_MAP = {
        "Performance": mlperf_loadgen.TestMode.PerformanceOnly,
        "Accuracy": mlperf_loadgen.TestMode.AccuracyOnly
    }

    parser = argparse.ArgumentParser()

    parser.add_argument('--scenario', type=str, default='SingleStream',
                            help="Scenario to be run: can be one of {SingleStream, Offline, MultiStream, Server}")

    parser.add_argument('--batch_size', type=int, default=32,
                            help="Max batch size to use in Offline and MultiStream scenarios.")

    parser.add_argument('--store_translation', default=False, action='store_true',
                            help="Store the output of translation? Note: Only valid with SingleStream scenario.")

    parser.add_argument('--verbose', default=False, action='store_true',
                            help="Verbose output.")

    parser.add_argument("--mode", default="Performance", help="Can be one of {Performance, Accuracy}")

    parser.add_argument("--debug_settings", default=False, action='store_true', 
        help="For debugging purposes, modify settings to small number of querries.")

    parser.add_argument("--qps", type=int, default=10, help="target qps estimate")

    parser.add_argument("--max-latency", type=str, default="0.100", help="mlperf max latency in 99pct tile")

    args = parser.parse_args()

    outdir = os.path.join(os.getcwd(), 'lg_output')

    # Create loadGen settings
    settings = mlperf_loadgen.TestSettings()
    try:
        settings.mode = MODE_MAP[args.mode]
        settings.scenario = SCENARIO_MAP[args.scenario]
    except KeyError as e:
        print("Unknown mode or scenario: {}".format(e.args[0]))
        raise e

    if args.qps:
        qps = float(args.qps)
        settings.server_target_qps = qps
        settings.offline_expected_qps = qps

    max_latency = [float(i) for i in args.max_latency.split(",")]

    # Specify input file
    if args.mode == "Accuracy":
        input_file = os.path.join(os.getcwd(), 'nmt', 'data', "newstest2014.tok.bpe.32000.en")
    else:
        input_file = os.path.join(os.getcwd(), 'nmt', 'data', "newstest2014.tok.bpe.32000.en.large")

    # Build the GNMT model
    if args.scenario == "SingleStream":
        batch_size = 1
    else:
        batch_size = args.batch_size

    gnmt_model = GNMTWrapper(batch_size = batch_size, outdir=outdir)

    if args.scenario == "SingleStream":
        runner = SingleStreamGNMTRunner(gnmt_model, input_file=input_file, store_translation=args.store_translation, verbose=args.verbose, outdir=outdir)
        
        # Specify exactly how many queries need to be made
        if args.debug_settings:
            settings.min_query_count = 80
            settings.max_query_count = 80

    elif args.scenario == "Offline":
        runner = GNMTRunner(gnmt_model, input_file=input_file, verbose=args.verbose)
        
        # Specify exactly how many queries need to be made
        if args.debug_settings:
            settings.min_query_count = 1
            settings.max_query_count = 1

    elif args.scenario == "MultiStream":
        runner = GNMTRunner(gnmt_model, input_file=input_file, verbose=args.verbose)
        
        # Specify exactly how many queries need to be made
        if args.debug_settings:
            settings.min_query_count = 100
            settings.max_query_count = 100
            settings.multi_stream_samples_per_query = 8

    elif args.scenario == "Server":
        runner = ServerGNMTRunner(gnmt_model, input_file=input_file, verbose=args.verbose)
        
        # Specify exactly how many queries need to be made
        if args.debug_settings:
            settings.min_query_count = 20
            settings.max_query_count = 100
    else:
        print("Invalid scenario selected")
        assert False

    # Create a thread in the GNMTRunner to start accepting work
    runner.start_worker()

    total_queries = runner.getTotalNumSentences() # Maximum sample ID + 1
    perf_queries = min(total_queries, 3003)   # Select the same subset of $perf_queries samples

    sut = mlperf_loadgen.ConstructSUT(runner.enqueue, flush_queries)
    qsl = mlperf_loadgen.ConstructQSL(
        total_queries, perf_queries, runner.load_samples_to_ram, runner.unload_samples_from_ram)


    # Start generating queries by starting the test
    # A single test for all non-server scenarios
    if not args.scenario == "Server":
        print("starting {} scenario".format(args.scenario))
        mlperf_loadgen.StartTest(sut, qsl, settings)

    # Multiple tests (depending on target latency array) for server scenario
    else:
        for target_latency in max_latency:
            print("starting {} scenario, latency={}".format(args.scenario, target_latency))
            settings.server_target_latency_ns = int(target_latency * NANO_SEC)

            mlperf_loadgen.StartTest(sut, qsl, settings)

    runner.finish()
    mlperf_loadgen.DestroyQSL(qsl)
    mlperf_loadgen.DestroySUT(sut)

