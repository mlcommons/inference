

class Model:
    def __init__(self):
        import tensorflow as tf
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)

    def load(self):
        import tensorflow as tf
        tf.saved_model.loader.load(self.session, ["serve"], "model-checkpoint")

        self.inputs = self.graph.get_tensor_by_name("Input/Input:0")
        self.labels = self.graph.get_tensor_by_name("Labels/Labels:0")
        self.loss   = self.graph.get_tensor_by_name(
            "Model/FullSoftmaxLoss_1_1/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:0")
        self.recurrentHidden = self.graph.get_tensor_by_name("Model/Placeholder:0")
        self.bias = self.graph.get_tensor_by_name("Model/Placeholder_1:0")

    def getLoss(self):
        return self.loss

    def getLabels(self):
        return self.labels

    def getInputs(self):
        return self.inputs

    def getRecurrentHidden(self):
        return self.recurrentHidden

    def getBias(self):
        return self.bias

    def run(self, inputs, labels):
        import numpy
        miniBatchSize = inputs.shape[0]

        return self.session.run(self.getLoss(),
            feed_dict={self.getInputs() : inputs,
                       self.getLabels() : labels,
                       self.getRecurrentHidden() : numpy.zeros((miniBatchSize, 2048)),
                       self.getBias() : numpy.zeros((miniBatchSize, 512))})

def getHashOfDirectoryTree(directory):
    import hashlib, os
    SHAhash = hashlib.md5()

    for root, dirs, files in os.walk(directory):
        for names in sorted(files):
            filepath = os.path.join(root,names)
            try:
                f1 = open(filepath, 'rb')
            except:
                # You can't open the file for some reason
                f1.close()
            continue

            while True:
                # Read file in as little chunks
                buf = f1.read(4096)
                if not buf : break
            SHAhash.update(hashlib.md5(buf).hexdigest())
            f1.close()

    return SHAhash.hexdigest()

def runOneBatch(model, dataset):
    inputs, labels = dataset.nextBatch()
    model.run(inputs, labels)

def runWarmup(model, dataset):
    print("Running warmup...")
    runOneBatch(model, dataset)
    dataset.reset()

def runBenchmarkWithTiming(arguments, model, dataset):
    import time

    print("Running benchmark...")

    losses = []
    times  = []

    for i in range(int(arguments["iterations"])):
        inputs, labels = dataset.nextBatch()

        start = time.time()
        losses.extend(model.run(inputs, labels))
        end = time.time()

        times.append(end - start)

    print("Longest latency was: " + str(sorted(times)[-1]) + " seconds.")
    print("Perplexity: " + str(getPerplexity(losses)) + ", target is 40.209 .")

def getPerplexity(losses):
    return 2**(sum(losses) / len(losses))

def runBenchmark(arguments, model, dataset):

    runWarmup(model, dataset)

    runBenchmarkWithTiming(arguments, model, dataset)


def extract(filename, path):
    import tarfile
    tar = tarfile.open(filename, "r:gz")
    tar.extractall(path)
    tar.close()

def downloadModel(arguments):
    import os
    import urllib.request
    filename = "model-checkpoint.tar.gz"
    if not os.path.exists(filename):
        print("Downloading model from " + arguments["model_url"])
        urllib.request.urlretrieve(arguments["model_url"], filename)

    print("extracting model " + filename)
    extract(filename, '.')

def loadModel():

    model = Model()

    model.load()

    return model

def compareChecksum(directory, checksum):
    import os
    if not os.path.exists(directory):
        print("directory does not exist: '" + directory + "'")
        return False

    computedChecksum = getHashOfDirectoryTree(directory)

    if computedChecksum != checksum:
        print("Checksum mismatch: '" + computedChecksum + "' vs " + "'" + checksum + "'")
        return False

    return True

class Dataset:
    def __init__(self, vocab, wordsPerSample, maximumSamples, miniBatchSize):
        self.vocab = vocab
        self.samples = []
        self.wordBuffer = []
        self.wordsPerSample = wordsPerSample
        self.maximumSamples = maximumSamples
        self.miniBatchSize = miniBatchSize
        self.index = 0

    def addWord(self, word):
        if self.isFull():
            return

        if not word in self.vocab:
            self.wordBuffer.append(self.vocab["UNKNOWN"])
        else:
            self.wordBuffer.append(self.vocab[word])

        if len(self.wordBuffer) > self.wordsPerSample:
            self.samples.append(self.wordBuffer)
            self.wordBuffer = []

    def isFull(self):
        return len(self.samples) >= self.maximumSamples

    def reset(self):
        self.index = 0

    def nextBatch(self):
        import numpy
        inputs = numpy.zeros((self.miniBatchSize, self.wordsPerSample + 1))

        for i in range(self.miniBatchSize):
            inputs[i, :] = self.nextSample()

        return inputs[:, 0:self.wordsPerSample], inputs[:, 1:]

    def nextSample(self):
        sample = self.samples[self.index]
        self.advance()
        return sample

    def advance(self):
        self.index += 1

def downloadValidationDataset(arguments):
    import os
    import urllib.request
    filename = "validation-dataset.tar.gz"
    if not os.path.exists(filename):
        print("Downloading dataset from " + arguments["validation_dataset_url"])
        urllib.request.urlretrieve(arguments["validation_dataset_url"], filename)

    print("extracting validation dataset " + filename)
    extract(filename, 'validation-dataset')

def loadVocab(arguments):
    import os
    path = arguments["vocab_path"]

    if not os.path.exists(path):
        raise RuntimeError("Could not locate validation data at: " + path)

    vocab = {}

    with open(path, "r") as vocabFile:
        for line in vocabFile:
            vocab[line.strip()] = len(vocab)

    vocab['UNKNOWN'] = len(vocab)

    return vocab

def loadValidationDataset(arguments):
    import os
    vocab = loadVocab(arguments)

    path = "validation-dataset/training-monolingual/news-commentary-v6.en"

    if not os.path.exists(path):
        raise RuntimeError("Could not locate validation data at: " + path)

    dataset = Dataset(vocab, int(arguments["words_per_sample"]), int(arguments["maximum_samples"]),
        int(arguments["mini_batch_size"]))

    with open(path, "r") as datasetFile:
        for line in datasetFile:
            for word in line.split():
                dataset.addWord(word)

                if dataset.isFull():
                    break

            if dataset.isFull():
                break

    return dataset

def getValidationDataset(arguments):
    if not compareChecksum("validation-dataset", arguments["validation_checksum"]):
        downloadValidationDataset(arguments)

    if not compareChecksum("validation-dataset", arguments["validation_checksum"]):
        raise RuntimeError("Checksum mismatch on validation dataset.")

    return loadValidationDataset(arguments)

def getModel(arguments):
    if not compareChecksum("model-checkpoint", arguments["model_checksum"]):
        downloadModel(arguments)

        if not compareChecksum("model-checkpoint", arguments["model_checksum"]):
            raise RuntimeError("Checksum mismatch on model.")
    else:
        print("Checksum for model passed, loading model...")

    return loadModel()

def main():
    from argparse import ArgumentParser

    parser = ArgumentParser(description="MLPerf Inference Language Modeling Benchmark.")

    parser.add_argument("-m", "--model-url",
        default = "https://zenodo.org/record/1492892/files/big-lstm.tar.gz?download=1",
        help = "Download the model from the specified url.")
    parser.add_argument("-I", "--iterations", default = 100,
        help = "The number of batches to run the inference benchmark for.")
    parser.add_argument("-b", "--mini-batch-size", default = 1,
        help = "The number of samples to process together in a batch.")
    parser.add_argument("--words-per-sample", default = 20,
        help = "The number of words in each sample.")
    parser.add_argument("--maximum-samples", default = 1000,
        help = "The number of samples to read from the validation dataset.")
    parser.add_argument("--model-checksum", default = "d41d8cd98f00b204e9800998ecf8427e",
        help = "The MD5 hash of the model.")
    parser.add_argument("-d", "--validation-dataset-url",
        default = "http://statmt.org/wmt11/training-monolingual-news-commentary.tgz",
        help = "Download the validation dataset from the specified url.")
    parser.add_argument("--vocab-path",
        default = "vocab.txt", help = "The list of words in the model's vocab")
    parser.add_argument("--validation-checksum", default = "d41d8cd98f00b204e9800998ecf8427e",
        help = "The MD5 hash of the validation dataset.")

    arguments = vars(parser.parse_args())

    model = getModel(arguments)
    validationDataset = getValidationDataset(arguments)

    runBenchmark(arguments, model, validationDataset)

################################################################################
## Guard Main
if __name__ == "__main__":
    main()
################################################################################




