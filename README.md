# Arborist Core

<img src="https://raw.githubusercontent.com/cmuarborist/cmuarborist.github.io/master/arborist-logo.jpg" height="150" align="right"/>

[https://cmuarborist.github.io/](https://cmuarborist.github.io/)

This repository contains the code for the Arborist method.

## Quickstart: CRIM on the Wordnet Taxonomy

Install required packages:
```
virtualenv --python=python2.7 env
source env/bin/activate
pip install -r requirements.txt
```

Create the Wordnet dataset:
```
python create_wordnet_dataset.py --data-folder data/wordnet/
```
This script does the following:
  - Retrieves the Wordnet graph via NLTK
  - Retains only edges of type `hypernym`, `substance_holonym` and `part_holonym`
  - Extracts the subgraph rooted at `animal.n.01`
  - Writes the subgraph to `data/mammal.csv` as `child, parent` pairs
  - Writes the nodes to `data/mammal.csv`

Get the embeddings of each node text.
You can use your favorite embedding method.
The following constructs FastText embeddings with the [Flair](https://github.com/flairNLP/flair) library:
```
virtualenv --python=python3 env3
source env3/bin/activate
pip3 install flair

wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip
unzip crawl-300d-2M-subword.zip

python generate_fasttext_embedding.py --vocab-file data/wordnet/mammal_vocab.csv
```

Create train/test splits of the taxonomy:

Further configuration options:
  - See `projection_learning_model/constants.py`
  - See `projection_learning_model/hyperparams.py`

## Contact

   * emaad@cmu.edu
