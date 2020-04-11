# Arborist Core

<img src="https://raw.githubusercontent.com/cmuarborist/cmuarborist.github.io/master/arborist-logo.jpg" height="150" align="right"/>

[https://cmuarborist.github.io/](https://cmuarborist.github.io/)

This repository contains the code for the Arborist taxonomy expansion method and the CRIM baseline.

## Quickstart: CRIM on the Wordnet Taxonomy

Install required packages:
```
virtualenv --python=python2.7 env
source env/bin/activate
pip install -r requirements_crim.txt
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
The following constructs FastText embeddings with the [Flair](https://github.com/flairNLP/flair) library
and stores them in `data/wordnet/mammal_vocab_fasttext_embeddings.csv`:
```
virtualenv --python=python3 env3
source env3/bin/activate # need a Python 3 environment
pip3 install flair

wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip
unzip crawl-300d-2M-subword.zip

python generate_fasttext_embedding.py --vocab-file data/wordnet/mammal_vocab.csv
```

Create train/test splits of the taxonomy (only leaf nodes are partitioned):
```
deactivate; source env/bin/activate # back in Python 2 environment
python create_train_test_split.py \
       --dataset data/wordnet/mammal.csv \
       --train-edges-file data/wordnet/mammal_train.csv \
       --test-edges-file data/wordnet/mammal_test.csv \
       --test-fraction 0.15
```

Create lists of the ancestors of each node:
```
python compute_ancestor_lists.py --edges-file data/wordnet/mammal_train.csv
```
This writes the ancestor lists to `data/wordnet/mammal_train_ancestors.csv` in the format `node,ancestor1,ancestor2,...`.

Train the model on `data/wordnet/mammal_train.csv`:
```
python -m projection_learning_model.train \
       --embeddings-file data/wordnet/mammal_vocab_fasttext_embeddings.csv \
       --train-edges-file data/wordnet/mammal_train.csv \
       --ancestors-file data/wordnet/mammal_train_ancestors.csv \
       --nprojections 48 --nnegs 5 --nepochs 10 \
       --batch-size 4096 --learning-rate 0.005 \
       --weight-decay 0.001 --dropout 0.01 \
       --save-folder crim_wordnet
```
This saves all model information in the `crim_wordnet` folder. Some useful diagnostics can be found as follows:
   - View train loss over epochs: `tail crim_wordnet/train_loss.txt`. The format is: `total_loss,positive_loss,negative_loss`
   - View validation loss over epochs: `tail crim_wordnet/train_loss.txt`. The format is: `total_loss,MRR,Recall@15,positive_loss,negative_loss`
   - Configure training and validation optionsin files `projection_learning_model/constants.py`, `projection_learning_model/hyperparams.py`.

Evaluate the model on `data/wordnet/mammal_test.csv`:
```
python -m projection_learning_model.evaluate \
       --embeddings-file data/wordnet/mammal_vocab_fasttext_embeddings.csv \
       --train-edges-file data/wordnet/mammal_train.csv \
       --test-edges-file data/wordnet/mammal_test.csv \
       --save-folder crim_wordnet \
       --model-file crim_wordnet/model.pytorch
```
The MRR, recall@15 and mean shortest-path-distance between predicted and actual parents is reported.
If `model-file` is not provided, the model with the highest validation MRR in `save-folder` is used.

## Contact

   * emaad@cmu.edu
