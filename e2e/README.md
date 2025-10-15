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

We extract all the unique wikipedia links, and download the web pages as PDFs or HTML. This is done using [`wkhtmltopdf`](https://wkhtmltopdf.org/) for PDFs or direct HTTP requests for HTML.

Additionally, it saves `url_mapping.json` which maps the url from FRAMES dataset and file name.

This will be used later for evaluation.

You may use [download_docs.py](./download_docs.py) script. 
```bash
$ python3 download_docs.py --help
usage: download_docs.py [-h] [--tsv_path TSV_PATH] [--max_urls MAX_URLS] [--output_dir OUTPUT_DIR]
                        [--output_data OUTPUT_DATA] [--processes PROCESSES] [--format {pdf,html}]

Download FRAMES dataset from Hugging Face and convert URLs to PDFs or HTML

options:
  -h, --help            show this help message and exit
  --tsv_path TSV_PATH   Input TSV file (default: download FRAMES dataset)
  --max_urls MAX_URLS   Maximum number of URLs to process (default: all)
  --output_dir OUTPUT_DIR
                        Output directory for downloaded documents (default: doc_pdf or doc_html)
  --output_data OUTPUT_DATA
                        Output directory for dataset file, if downloaded from Hugging Face (default:
                        data)
  --processes PROCESSES
                        Number of parallel processes (default: 10)
  --format {pdf,html}   Output format: pdf or html (default: pdf)

## Sample usage
$ python3 download_docs.py --output_dir doc_pdf --format pdf --processes 30 
$ python3 download_docs.py --output_dir doc_html --format html --processes 30
## Will download documents in ./doc_pdf/ or ./doc_html/
```

## Corpus preprocessing
We now need to extract text content from the set of PDFs or HTML files. This is done using [PyMuPDF](https://pypi.org/project/PyMuPDF/) for PDFs or [BeautifulSoup4](https://pypi.org/project/beautifulsoup4/) for HTML.

Important considerations:
- Extracted text from a single document has many characters.
- However, embedding models (especially rerankers like [`ColBERTv2`](https://huggingface.co/colbert-ir/colbertv2.0)) have a size restriction on the maximum length of sequence they can encode (~512). 
- Thus, we break down documents into chunks. We call these chunks "passages".
    - Each passage belongs to a single unique document source.

In this step:
- our input is a set of PDFs or HTML documents
- our output is:
    - a set of txt documents (1 per source), and
    - a JSON file of passages.

The passages JSON has a schema as follows:
```json
{
    "index": 0,
    "base_filename": "name_of_file",
    "original_url": "https://...",
    "passage": "Long passage from (part of) document"
}
```

Here, original_url is from `url_mapping.json` and will be used to grade if the correct document has been retrieved.

You may use the [`read_docs.py`](./read_docs.py) script. 
```bash
$ python3 read_docs.py --help
usage: read_docs.py [-h] [--json JSON] [--max-files MAX_FILES] [--max-length MAX_LENGTH]
                    [--overlap OVERLAP] [--fixed-length FIXED_LENGTH] [--fixed-overlap FIXED_OVERLAP]
                    [--no-tables] [--no-lists] [--text-boundary {sentence,word,none}] [--benchmark]
                    input_dir output_dir

Extract text from PDF and HTML files and split into passages

positional arguments:
  input_dir             Directory containing document files (PDF/HTML)
  output_dir            Directory to save text files

options:
  -h, --help            show this help message and exit
  --json JSON           JSON file to save passage data
  --max-files MAX_FILES
                        Maximum number of files to process (for testing)
  --max-length MAX_LENGTH
                        Maximum passage length in characters (default: 512)
  --overlap OVERLAP     Overlap between passages in characters (default: 50)
  --fixed-length FIXED_LENGTH
                        Use fixed-length passages instead of variable-length
  --fixed-overlap FIXED_OVERLAP
                        Overlap for fixed-length passages (default: 32)
  --no-tables           Don't preserve table structure (HTML only)
  --no-lists            Don't preserve list structure (HTML only)
  --text-boundary {sentence,word,none}
                        Text boundary optimization (default: sentence)
  --benchmark           Enable performance monitoring

## Sample usage
$ python3 read_docs.py doc_pdf passages --max-length 256 --json passages/doc_pdf_len256.json --overlap 32
$ python3 read_docs.py doc_html passages --max-length 256 --json passages/doc_html_len256.json --overlap 32
```

### TODO Items for passage chunking:
~~1. Chunking is done at a character level right now. We may need to do this at the token level directly to avoid truncation.~~

Use --text-boundary

~~2. Need to assess the quality of text extraction - PyMuPDF may have ordering issues, HTML parsing needs filtering of non-content elements.~~

HTML has ~1% better accuracy than PDF-processed passages.

3. Passage size will directly affect vector operations. Need to study impact of passage len + overlap on vector size, ingestion time, lookup time, etc.

## Vector Database & Indexing

The retrieval system supports multiple backends and index types:

### Retrieval Methods
- **BM25**: Sparse lexical retrieval (traditional keyword search)
- **Vector**: Dense semantic retrieval using embeddings

### Vector Index Methods
- **Flat L2**: Exact k-NN search, O(n) time, best for <10K documents
- **HNSW**: Approximate search, O(log n) time, balanced accuracy & latency, default choice
- **IVF**: Inverted file index, memory efficient, best for >1M documents
  - Configurable `nprobe` parameter (default: 10) controls accuracy/speed tradeoff (higher the better accuracy)

## Single-shot retrieval & evaluation

Use [`single_shot_retrieval.py`](./single_shot_retrieval.py) to ingest passages, perform retrieval, and evaluate accuracy.

```bash
# Ingest passages and save database
$ python3 single_shot_retrieval.py --ingest passages/data.json --db my_db --retrieval_method vector

# Load existing database and run single query
$ python3 single_shot_retrieval.py --db my_db.db --retrieval_method vector --query "Your question"

# Run full evaluation on FRAMES dataset
$ python3 single_shot_retrieval.py --db my_db.db --retrieval_method vector --eval
$ python3 single_shot_retrieval.py --db my_db.db --retrieval_method bm25 --bm25_stemmer pystemmer --eval

# Key options:
#   --device                                 Device to run the models on (e.g., 'cpu', 'cuda', 'xpu', or 'auto')
#   --retrieval_strategy                     Retrieval strategy: 'fixed_k' (default), 'top_p' (nucleus sampling), 'relative' (score-based)
#   --retrieval_method {bm25,vector}         Retrieval backend
#   --vector_index_method {flat,hnsw,ivf}    Vector index type (default: hnsw)
#   --eval [N]                               Evaluate accuracy using N queries from dataset specified by --dataset 
#   --ivf_nprobe N                           IVF clusters to search (default: 10)
#   --top_k_retriever N                      Top-K documents to retrieve (default: 25)
#   --top_k_reranking N                      Top-K after reranking (default: 10)
#   --no-rerank                              Skip reranking step
#   --no-save                                Not saving DB for testing purpose
#   --benchmark                              Enable performance monitoring
...
```

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