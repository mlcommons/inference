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
cd /work && ./setup.sh
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

You may use the [`read_pdf.py`](./read_pdf.py) script. 
```bash
$ python3 read_pdf.py --help
usage: read_pdf.py [-h] [--json-file JSON_FILE] [--max-files MAX_FILES] [--max-length MAX_LENGTH]
                   [--overlap OVERLAP]
                   input_dir output_dir

Extract text from all PDFs in a directory

positional arguments:
  input_dir             Input directory containing PDF files
  output_dir            Output directory for text files

options:
  -h, --help            show this help message and exit
  --json-file JSON_FILE
                        Output JSON file path for passages data (enables JSON creation)
  --max-files MAX_FILES
                        Maximum number of PDF files to process (default: all files)
  --max-length MAX_LENGTH
                        Maximum length of each passage in characters (default: 512)
  --overlap OVERLAP     Overlap between passages in characters (default: 50)


## Sample usage
$ python3 read_pdf.py doc_pdf doc_txt_len256_overlap32 --max-length 256 --json-file doc_txt_fixed_len256_overlap32/passages.json --overlap 32
```

### TODO Items for passage chunking:
1. Chunking is done at a character level right now. We may need to do this at the token level directly to avoid truncation.
2. Need to assess the quality of text extraction from the PyMuPDF package - right now, it seems like some texts are jumbled in order. 
3. Passage size will directly affect vector operations. Need to study impact of passage len + overlap on vector size, ingestion time, lookup time, etc.

## Single-shot lookup
1. Embed query: `Query text` -> `Query Tokens` -> `Query vector`
2. Perform vector similarity search, and return top-k documents. 
3. Perform reranking using ColBERT (Late interaction and `MaxSim` scoring)

Rerankers: Slow but accurate  
Retrievers: Fast but less accurate

## Multi-step lookup (TODO)
Instead of a single step retrieval, we perform multiple steps. In each step, we give the LLM partial retrieved context, and the user query - and ask it to generate search queries. This helps in breaking down multi-step reasoning questions.

Consider, as an example, the below query: 
```none
Who won the French Open Mens Singles tournament the year that New York City FC won their first MLS Cup title?
```

This is a classic multi-step reasoning. The logical deduction of a well-performing system is: 
```
- What year did New York City FC win their first MLS Cup title
(retrieve docs regarding MLS cup winners)
(say, answer is 2005)
- Who won the French Open Mens Singles tournament in 2005?
(retrieve 2005 French open document)
```

The flow now looks something like: 
1. User query comes in
2. Repeat 1..n times:  
    1. Given to query rewriter, which gives at most k search queries.
    2. For each query:
        1. Encode into vector
        2. Perform vector search and retrieve relevant documents
3. Rerank retrieved documents, and choose top-n (call this filtered documents)
4. Give filtered documents + user query to LLM generator

The query rewriter may be thought of as an LLM with the following prompt: 
```
You are an expert at generating search queries to help answer complex questions using a collection of Wikipedia articles. 

Given the following:
- The user's original question.
- Relevant facts or documents already gathered so far (if any).

Your task:  
Generate [k] concise, focused search queries that could be used to find specific information from Wikipedia to help answer the question.  
- Make each query target a different aspect of the problem or missing information.  
- Avoid duplicating information already in the context.  
- Do not reference source filenames, document titles, or include any special characters.
- Think step by step before writing each query.
- List the missing pieces of information, then write k queries that could best retrieve them.

[User Question:]
{user_question}

[Known Facts / Retrieved Documents:]
{summarized_partial_context}

```