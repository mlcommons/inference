#!/bin/bash
mkdir Datasets
mkdir Datasets/Clean_IMDB

echo "### Downloading 'Large Movie Review Dataset' from - http://ai.stanford.edu/~amaas/data/sentiment/"

curl http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz | tar -xz  -C Datasets/Clean_IMDB --strip-components 1


