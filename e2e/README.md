# E2E: RAG benchmark
This is a WIP proposal, and will undergo changes.

## Benchmark flow
We start with a corpus of documents and a set of user queries. 

First, we preprocess the corpus to get a vector DB.

Then, given a user query, we perfrom:
1. retrieval
2. reranking
3. answer generation

## Preliminary: Environment setup
The recommended environment setup depends on enroot installation - however, it is simple enough that docker may be used. 

### Base environment
It is assumed that the user starts from a Ubuntu PyTorch base image (sandbox/container)

You may use either:
- [Ubuntu base image from dockerhub](https://hub.docker.com/_/ubuntu) (not recommended)
- [NVIDIA PyTorch image from NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags) (recommended)
- [AMD ROCm/PyTorch image from dockerhub](https://hub.docker.com/r/rocm/pytorch/tags) (not tested yet)

### Enroot setup
If you have enroot, you can get started by:
```bash
mkdir -p containers/

enroot import -o containers/my_image.sqsh dockerd://docker_image:tag 
# For example, docker://pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

enroot create --name my_sandbox containers/my_image.sqsh

enroot start --root --rw \
    --mount /actual/path:/mounted/path \
    --mount $(pwd):/work my_sandbox

## Once inside sandbox, install dependancies via
cd /work && ./setup.cfg
```

## Corpus creation
Starting from [FRAMES](https://huggingface.co/datasets/google/frames-benchmark), we have a set of tuples as:
```
[UserQuery, WikiLinks, Answer]
```

We extract all the unique wikipedia links, and download the web pages as PDFs. This is done using [`wkhtmltopdf`](https://wkhtmltopdf.org/).

You may use [download_pdf.py](./download_pdf.py) script. 
```bash
$ python3 download_pdf.py --help
usage: download_pdf.py [-h] [--tsv_path TSV_PATH] [--max_urls MAX_URLS] [--output_pdf OUTPUT_PDF]
                       [--output_data OUTPUT_DATA] [--processes PROCESSES]

Download FRAMES dataset from Hugging Face and convert URLs to PDFs

options:
  -h, --help            show this help message and exit
  --tsv_path TSV_PATH   Input TSV file (default: download FRAMES dataset)
  --max_urls MAX_URLS   Maximum number of URLs to process (default: all)
  --output_pdf OUTPUT_PDF
                        Output directory for PDFs (default: doc_pdf)
  --output_data OUTPUT_DATA
                        Output directory for dataset file, if downloaded from Hugging Face (default:
                        data)
  --processes PROCESSES
                        Number of parallel processes (default: 10)

## Sample usage
$ python3 download_pdf.py --output_pdf doc_pdf_fixed --processes 30 
## Will download PDFs in ./doc_pdf/
```

## Corpus preprocessing
We now need to extract text content from the set of PDFs. This is done using the [PyMuPDF](https://pypi.org/project/PyMuPDF/) package (also referred to as `fitz`)

Important considerations:
- Extracted text from a single PDF document has many characters.
- However, embedding models (especially rerankers like [`ColBERTv2`](https://huggingface.co/colbert-ir/colbertv2.0)) have a size restriction on the maximum length of sequence they can encode (~512). 
- Thus, we break down a single PDFs into chunks. We call these chunks "passages".
    - Each passage belongs to a single unique PDF source.

In this step:
- our input is a set of PDFs
- our output is:
    - a set of txt documents (1 per PDF), and
    - a JSON file of passages.

The passages JSON has a schema as follows:
```json
{
    "index": 0,
    "pdf_filename": "name_of_file.pdf",
    "passage": "Long passage from (part of) pdf_filename"
}
```

### TODO Items for passage chunking:
1. Chunking is done at a
