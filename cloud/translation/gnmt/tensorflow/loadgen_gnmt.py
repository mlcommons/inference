from generic_loadgen import *
import sys
from nmt.nmt import create_hparams, add_arguments, create_or_load_hparams
import argparse
from nmt.inference import inference as inference_fn
import tensorflow as tf
import os
from nmt.utils import misc_utils as utils

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

    # TBD: clean this up!
    def setup(self, flags, default_hparams):
        # Random
        random_seed = flags.random_seed
        if random_seed is not None and random_seed > 0:
          utils.print_out("# Set random seed to %d" % random_seed)
          random.seed(random_seed + jobid)
          np.random.seed(random_seed + job)

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
              save_hparams=(jobid == 0))

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

        # Inference
        ckpt = flags.ckpt
        if not ckpt:
            ckpt = tf.train.latest_checkpoint(out_dir)

        self.hparams = default_hparams
        self.ckpt = ckpt
        self.runmode = flags.run


    ##
    # @brief Invoke GNMT to translate the input file
    def process(self, qitem):
        print("translate {} (QID {}): {} --> {}".format(self.count, qitem.query_id, qitem.input_file, qitem.output_file))
        input_file = qitem.input_file 
        output_file = qitem.output_file
        # TBD: replace this by single_worker inference fn
        inference_fn(self.runmode, 1, self.ckpt, input_file,
                    output_file, self.hparams, num_workers=1, jobid=-1)
        
        self.count += 1
        
        return self.count

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

    print ("Starting pool")

    settings = mlperf_loadgen.TestSettings()
    settings.scenario = mlperf_loadgen.TestScenario.SingleStream
    settings.mode = mlperf_loadgen.TestMode.PerformanceOnly
    settings.samples_per_query = 1
    settings.target_qps = 10        # Doesn't seem to have an effect
    settings.target_latency_ns = 1000000000

    
    total_queries = 256 # Maximum sample ID + 1
    perf_queries = 8   # TBD: Doesn't seem to have an effect

    sut = mlperf_loadgen.ConstructSUT(runner.enqueue, process_latencies)
    qsl = mlperf_loadgen.ConstructQSL(
        total_queries, perf_queries, load_samples_to_ram, unload_samples_from_ram)
    mlperf_loadgen.StartTest(sut, qsl, settings)
    mlperf_loadgen.DestroyQSL(qsl)
    mlperf_loadgen.DestroySUT(sut)

