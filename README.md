# Arborist Core

<img src="https://raw.githubusercontent.com/cmuarborist/cmuarborist.github.io/master/arborist-logo.jpg" height="150" align="right"/>

[https://cmuarborist.github.io/](https://cmuarborist.github.io/)

This repository contains the code for the Arborist taxonomy expansion method and the [CRIM](https://www.aclweb.org/anthology/S18-1116/) baseline.

To test if your embeddings (or, more generally, your node feature vectors) have enough signal to predict taxonomy parents, we recommend running CRIM first (since it is a simpler model, and easier/faster to train).

If CRIM performs reasonably well, Arborist will further improve performance by exploitng the latent heterogenous edge semantics (if any) present in the taxonomy.

## Quickstart: Arborist on the Wordnet Taxonomy

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
and stores them in `data/wordnet/mammal_vocab_fasttext_embeddings.csv` as `word,embedding_dim1,embedding_dim2,...`:
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

Create lists of the ancestors of each node (so they can be excluded from the negative samples for each node):
```
python compute_ancestor_lists.py --edges-file data/wordnet/mammal_train.csv
```
This writes the ancestor lists to `data/wordnet/mammal_train_ancestors.csv` in the format `node,ancestor1,ancestor2,...`.

Compute shortest-path distances between all pairs of nodes in the training taxonomy subset:
```
python compute_shortest_path_distances.py --edges-file data/wordnet/mammal_train.csv --embeddings-file data/wordnet/mammal_vocab_fasttext_embeddings.csv
```
This writes the shortest-path distances to `data/wordnet/mammal_train_spdist.npy`. These are used to construct the margins in the large-margin loss.

Train the model on `data/wordnet/mammal_train.csv`:
```
python -m large_margin_taxonomic_role_model.train \
       --embeddings-file data/wordnet/mammal_vocab_fasttext_embeddings.csv \
       --train-edges-file data/wordnet/mammal_train.csv \
       --ancestors-file data/wordnet/mammal_train_ancestors.csv \
       --spdist-file data/wordnet/mammal_train_spdist.npy \
       --nprojections 48 --nnegs 5 --nepochs 10 \
       --batch-size 4096 --learning-rate 0.0005 \
       --weight-decay 0.001 --dropout 0.01 --upweight 0.25\
       --save-folder arborist_wordnet
```
This saves all model information in the `arborist_wordnet` folder. Some useful diagnostics can be found as follows:
   - View train loss over epochs: `tail arborist_wordnet/train_loss.txt`. The format is: `total_loss,positive_loss,negative_loss`
   - View validation loss over epochs: `tail arborist_wordnet/train_loss.txt`. The format is: `total_loss,MRR,Recall@15,positive_loss,negative_loss`
   - Configure training and validation options in files `large_margin_taxonomic_role_model/constants.py`, `large_margin_taxonomic_role_model/hyperparams.py`.

Evaluate the model on `data/wordnet/mammal_test.csv`:
```
python -m large_margin_taxonomic_role_model.evaluate \
       --nprojections 48 \
       --embeddings-file data/wordnet/mammal_vocab_fasttext_embeddings.csv \
       --train-edges-file data/wordnet/mammal_train.csv \
       --test-edges-file data/wordnet/mammal_test.csv \
       --save-folder arborist_wordnet \
       --model-file arborist_wordnet/model.pytorch
```
The MRR, recall@15 and mean shortest-path-distance between predicted and actual parents is reported.
Predictions are made by computing cosine distances to internal training nodes only (since leaf nodes cannot have children).
If `model-file` is not provided, the model with the highest validation MRR in `save-folder` is used.

## Quickstart: CRIM on the Wordnet Taxonomy

This is a slightly improved version of CRIM with the same enhancements used for Arborist.

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
and stores them in `data/wordnet/mammal_vocab_fasttext_embeddings.csv` as `word,embedding_dim1,embedding_dim2,...`:
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

Create lists of the ancestors of each node (so they can be excluded from the negative samples for each node):
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
Predictions are made by computing cosine distances to internal training nodes only (since leaf nodes cannot have children).
If `model-file` is not provided, the model with the highest validation MRR in `save-folder` is used.

## Contact

   * emaad@cmu.edu
