from generic_loadgen import *
import sys
from nmt.nmt import create_hparams, add_arguments, create_or_load_hparams
import argparse
from nmt.inference import _decode_inference_indices, get_model_creator, start_sess_and_load_model, load_data
import tensorflow as tf
import os
from nmt.utils import misc_utils as utils
from nmt.utils import nmt_utils
from nmt import model_helper
import codecs

class TranslationTask:
    def __init__(self, query_id, input_file, output_file):
        self.query_id = query_id
        self.input_file = input_file
        self.output_file = output_file
        self.start = time.time()

class GNMTRunner (Runner):
    def __init__(self):
        Runner.__init__(self)
            
        ckpt_path = os.path.join(os.getcwd(), 'ende_gnmt_model_4_layer',
                        'translate.ckpt')
        outdir = os.path.join(os.getcwd(), 'nmt', 'data', 'result', 'output')
        hparams_path= os.path.join(os.getcwd(), 'nmt', 'standard_hparams',
                             'wmt16_gnmt_4_layer.json')
        vocab_prefix = os.path.join(os.getcwd(), 'nmt', 'data', 'vocab.bpe.32000')

        flags, h_params = self.setup_params_and_flags(ckpt_path, hparams_path, outdir, vocab_prefix)

        self.setup(flags, h_params)

        #inference_fn = inference.inference
        #run_main(FLAGS, default_hparams, train_fn, inference_fn)

        print (self.runmode)

        Runner.__init__(self)
        self.count = 0

    def setup_params_and_flags(self, ckpt_path, hparams_path, outdir, vocab_prefix):
        FLAGS = None
        # TBD remove argument parsing, and just have it return all default values.
        nmt_parser = argparse.ArgumentParser()
        add_arguments(nmt_parser)
        FLAGS, unparsed = nmt_parser.parse_known_args()

        # Some of these flags are never used and are just set for consistency
        FLAGS.num_workers = 1
        FLAGS.iterations = 1
        FLAGS.infer_batch_size = 1    # SingleStream scenario
        FLAGS.num_inter_threads = 1
        FLAGS.num_intra_threads = 1
        FLAGS.run = "accuracy" # Needs to be set to accuracy to generate output
        # Pass in inference specific flags
        # TBD: parametrize
        FLAGS.ckpt = ckpt_path
        FLAGS.src = 'en'
        FLAGS.tgt = 'de'
        FLAGS.hparams_path = hparams_path
        FLAGS.out_dir = outdir
        FLAGS.vocab_prefix = vocab_prefix

        default_hparams = create_hparams(FLAGS)
        return FLAGS, default_hparams

    def setup(self, flags, default_hparams):
        # Model output directory
        out_dir = flags.out_dir
        if out_dir and not tf.gfile.Exists(out_dir):
          utils.print_out("# Creating output directory %s ..." % out_dir)
          tf.gfile.MakeDirs(out_d)

        # Load hparams.
        loaded_hparams = False
        if flags.ckpt:  # Try to load hparams from the same directory as ckpt
          ckpt_dir = os.path.dirname(flags.ckpt)
          ckpt_hparams_file = os.path.join(ckpt_dir, "hparams")
          if tf.gfile.Exists(ckpt_hparams_file) or flags.hparams_path:
                hparams = create_or_load_hparams(ckpt_dir, default_hparams, flags.hparams_path, save_hparams=False)
                loaded_hparams = True
        if not loaded_hparams:  # Try to load from out_dir
          assert out_dir
          hparams = create_or_load_hparams(out_dir, default_hparams, flags.hparams_path,
              save_hparams = True)

        # GPU device
        config_proto = utils.get_config_proto(
            allow_soft_placement=True,
            num_intra_threads=hparams.num_intra_threads,
            num_inter_threads=hparams.num_inter_threads)
        utils.print_out(
            "# Devices visible to TensorFlow: %s" 
            % repr(tf.Session(config=config_proto).list_devices()))

        # Inference indices
        hparams.inference_indices = None
        if flags.inference_list:
            (hparams.inference_indices) = ([int(token)  for token in flags.inference_list.split(",")])

        

        model_creator = get_model_creator(hparams)
        infer_model = model_helper.create_infer_model(model_creator, hparams, scope=None)
        sess, loaded_infer_model = start_sess_and_load_model(infer_model, flags.ckpt,
                                                       hparams)

        #tbd: clean up
        self.hparams = hparams
        self.ckpt = flags.ckpt
        self.runmode = flags.run

        self.infer_model = infer_model
        self.sess = sess
        self.loaded_infer_model = loaded_infer_model

    ##
    # @brief Invoke GNMT to translate the input file
    def process(self, qitem):
        print("translate {} (QID {}): {} --> {}".format(self.count, qitem.query_id, qitem.input_file, qitem.output_file))
        input_file = qitem.input_file 
        trans_file = qitem.output_file

        # FIXME: loaod only needed data here.
        input_file =  os.path.join(os.getcwd(), 'nmt', 'data', 'newstest2014.tok.bpe.32000.en')
        infer_data = load_data(input_file, self.hparams)
        query_index = 3

        infer_mode = self.hparams.infer_mode
        num_translations_per_input = self.hparams.num_translations_per_input

        # Set input data and batch size
        with self.infer_model.graph.as_default():
            self.sess.run(
                self.infer_model.iterator.initializer,
                feed_dict={
                    self.infer_model.src_placeholder: [infer_data[query_index]],
                    self.infer_model.batch_size_placeholder: self.hparams.infer_batch_size
                })


        # Start the translation
        nmt_outputs, _ = self.loaded_infer_model.decode(self.sess)
        if infer_mode != "beam_search":
          nmt_outputs = np.expand_dims(nmt_outputs, 0)

        # SingleStream means we are only processing one batch, make sure this is the case
        assert self.hparams.infer_batch_size == nmt_outputs.shape[1] == 1

        # Whether beam search is being used or not, we only want 1 final translation
        assert self.hparams.num_translations_per_input == 1
        sent_id = 0 # Sinds there is only one sample in the batch

        translation = nmt_utils.get_translation(
                    nmt_outputs[0],
                   sent_id,
                   tgt_eos=self.hparams.eos,
                   subword_option=self.hparams.subword_option)
            #TBD: some code tto write out translation. Move this somewhere.
        """
          with codecs.getwriter("utf-8")(
              tf.gfile.GFile(trans_file, mode="wb")) as trans_f:
            trans_f.write((translation + b"\n").decode("utf-8"))
        """
            
        self.count += 1
        
        return translation    
    
    ##
    # @brief Create a task and add it to the queue
    def enqueue(self, query_samples):
        for sample in query_samples:
            # TBD parametrize
            input_file = os.path.join("loadgen_files", "in_{}".format(sample.index))
            output_file = os.path.join("lg_output", "out_{}".format(sample.index))

            task = TranslationTask(sample.id, input_file, output_file)
            self.tasks.put(task)

if __name__ == "__main__":
    runner = GNMTRunner()

    runner.start_worker()

    settings = mlperf_loadgen.TestSettings()
    settings.scenario = mlperf_loadgen.TestScenario.SingleStream
    settings.mode = mlperf_loadgen.TestMode.PerformanceOnly
    settings.samples_per_query = 1
    settings.target_qps = 10        # Doesn't seem to have an effect
    settings.target_latency_ns = 1000000000

    
    total_queries = 3003 # Maximum sample ID + 1
    perf_queries = 3003   # Select the same subset of $perf_queries samples

    sut = mlperf_loadgen.ConstructSUT(runner.enqueue, process_latencies)
    qsl = mlperf_loadgen.ConstructQSL(
        total_queries, perf_queries, load_samples_to_ram, unload_samples_from_ram)
    mlperf_loadgen.StartTest(sut, qsl, settings)
    mlperf_loadgen.DestroyQSL(qsl)
    mlperf_loadgen.DestroySUT(sut)

