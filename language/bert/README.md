# MLPerf Inference Benchmarks for Natural Language Processing

This is the reference implementation for MLPerf Inference benchmarks for Natural Language Processing.

The chosen model is BERT-Large performing SQuAD v1.1 question answering task.

Please see [this readme](README_cm.md) file for an automated way to run this benchmark out of the box and do an end-to-end submission with or without docker using the [MLCommons CM](https://github.com/mlcommons/ck/tree/master/cm) language.

## Prerequisites

- [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
- Any NVIDIA GPU supported by TensorFlow or PyTorch

## Supported Models

| model | framework | accuracy | dataset | model link | model source | precision | notes |
| ----- | --------- | -------- | ------- | ---------- | ------------ | --------- | ----- |
| BERT-Large | TensorFlow | f1_score=90.874% | SQuAD v1.1 validation set | [from zenodo](https://zenodo.org/record/3733868) [from zenodo](https://zenodo.org/record/3939747) | [BERT-Large](https://github.com/google-research/bert), trained with [NVIDIA DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT) | fp32 | |
| BERT-Large | PyTorch | f1_score=90.874% | SQuAD v1.1 validation set | [from zenodo](https://zenodo.org/record/3733896) | [BERT-Large](https://github.com/google-research/bert), trained with [NVIDIA DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT), converted with [bert_tf_to_pytorch.py](bert_tf_to_pytorch.py) | fp32 | |
| BERT-Large | ONNX | f1_score=90.874% | SQuAD v1.1 validation set | [from zenodo](https://zenodo.org/record/3733910) | [BERT-Large](https://github.com/google-research/bert), trained with [NVIDIA DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT), converted with [bert_tf_to_pytorch.py](bert_tf_to_pytorch.py) | fp32 | |
| BERT-Large | ONNX | f1_score=90.067% | SQuAD v1.1 validation set | [from zenodo](https://zenodo.org/record/3750364) | Fine-tuned based on the PyTorch model and converted with [bert_tf_to_pytorch.py](bert_tf_to_pytorch.py) | int8, symetrically per-tensor quantized without bias | See [MLPerf INT8 BERT Finetuning.pdf](MLPerf INT8 BERT Finetuning.pdf) for details about the fine-tuning process |
| BERT-Large | PyTorch | f1_score=90.633% | SQuAD v1.1 validation set | [from zenodo](https://zenodo.org/record/4792496) | Fine-tuned based on [Huggingface bert-large-uncased pretrained model](https://huggingface.co/bert-large-uncased) | int8, symetrically per-tensor quantized without bias | See README.md at Zenodo link for details about the fine-tuning process |

## Disclaimer
This benchmark app is a reference implementation that is not meant to be the fastest implementation possible.

## Commands

Please run the following commands:

- `make setup`: initialize submodule, download datasets, and download models.
- `make build_docker`: build docker image.
- `make launch_docker`: launch docker container with an interaction session.
- `python3 run.py --backend=[tf|pytorch|onnxruntime|tf_estimator] --scenario=[Offline|SingleStream|MultiStream|Server] [--accuracy] [--quantized]`: run the harness inside the docker container. Performance or Accuracy results will be printed in console.

## Details

- SUT implementations are in [tf_SUT.py](tf_SUT.py), [tf_estimator_SUT.py](tf_estimator_SUT.py) and [pytorch_SUT.py](pytorch_SUT.py). QSL implementation is in [squad_QSL.py](squad_QSL.py).
- The script [accuracy-squad.py](accuracy-squad.py) parses LoadGen accuracy log, post-processes it, and computes the accuracy.
- Tokenization and detokenization (post-processing) are not included in the timed path.
- The inputs to the SUT are `input_ids`, `input_make`, and `segment_ids`. The output from SUT is `start_logits` and `end_logits` concatenated together.
- `max_seq_length` is 384.
- The script [tf_freeze_bert.py] freezes the TensorFlow model into pb file.
- The script [bert_tf_to_pytorch.py] converts the TensorFlow model into the PyTorch `BertForQuestionAnswering` module in [HuggingFace Transformers](https://github.com/huggingface/transformers) and also exports the model to [ONNX](https://github.com/onnx/onnx) format.

## License

Apache License 2.0
