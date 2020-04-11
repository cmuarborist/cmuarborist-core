# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function

import argparse
import networkx as nx
import numpy as np
import projection_learning_model.model as projection_learning_model

if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    # files
    ap.add_argument("--embeddings-file", required=True)
    ap.add_argument("--train-edges-file", required=True)
    ap.add_argument("--ancestors-file", required=True)

    # hyperparameters
    ap.add_argument("--nprojections", required=True, default=24,
                    help="Number of projections")
    ap.add_argument("--nnegs", required=False, default=10,
                    help="Number of negative samples")
    ap.add_argument("--nepochs", required=False, default=50,
                    help="Number of training epochs")
    ap.add_argument("--dropout", required=False, default=0.0,
                    help="Dropout probability")
    ap.add_argument("--learning-rate", required=False, default=0.01,
                    help="Learning rate")
    ap.add_argument("--batch-size", required=False, default=32,
                    help="Training batch size")
    ap.add_argument("--grad-clip", required=False, help="Maximum gradient norm")
    ap.add_argument("--weight-decay", required=False, default=0,
                    help="Weight decay factor")
    ap.add_argument("--save-folder", required=True,
                    help="Folder to store the trained model and losses")

    args = ap.parse_args()
    embeddings_file = args.embeddings_file
    train_edges_file = args.train_edges_file
    ancestors_file = args.ancestors_file

    nprojections = int(args.nprojections)
    nnegs = int(args.nnegs)
    nepochs = int(args.nepochs)
    batch_size = int(args.batch_size)
    learning_rate = float(args.learning_rate)
    weight_decay = float(args.weight_decay)
    dropout = float(args.dropout)
    save_folder = args.save_folder

    if not args.grad_clip:
        grad_clip = np.inf  # no clipping by default
    else:
        grad_clip = float(args.grad_clip)

    # Load embeddings
    row_idx = 0
    embeddings = []
    topicid2row = {}  # maps each topic ID to an embedding matrix row index
    dag = nx.Graph()
    with open(embeddings_file, "r") as f:
	for line in f:
	    fields = line.strip().split(",")
	    topicid = fields[0]
	    vector = map(float, fields[1:])

	    topicid2row[topicid] = row_idx
	    embeddings.append(vector)
	    dag.add_node(row_idx)

	    row_idx += 1
    embeddings = np.array(embeddings)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) # normalize to unit length

    # Load train edges 
    train_edges = []  # row indices of child-parent pairs
    parent_sets = {}  # compatability for multiple parents

    with open(train_edges_file, "r") as f:
	for line in f:
	    child_topicid, parent_topicid = line.strip().split(",")
	    child_idx, parent_idx = topicid2row[child_topicid], topicid2row[parent_topicid]
	    train_edges.append((child_idx, parent_idx))
	    parent_sets[child_idx] = set([parent_idx])
	    dag.add_edge(child_idx, parent_idx)
    train_edges = np.array(train_edges)

    # Load ancestors
    ancestor_edges = []
    with open(ancestors_file, "r") as f:
	for line in f:
	    fields = line.strip().split(",")
	    if len(fields[1]) == 0:
		continue
	    row_ids = [topicid2row[f] for f in fields]
	    child_idx = row_ids[0]
	    for ancestor_idx in row_ids[1:]:
		ancestor_edges.append((child_idx, ancestor_idx))
    ancestor_edges = np.array(ancestor_edges)

    # Train model
    model = projection_learning_model.NeuralTaxonomyExpander(
		embeddings=embeddings, num_projections=nprojections,
		num_negative_samples=nnegs, dropout_prob=dropout)

    model.train() # put pytorch model in train mode
    model.fit(train_edges, parent_sets, ancestor_edges=ancestor_edges,
	      batch_size=batch_size, nepochs=nepochs, save_folder=save_folder,
	      learning_rate=learning_rate, weight_decay=weight_decay)
