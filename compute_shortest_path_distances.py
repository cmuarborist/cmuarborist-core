# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function

import argparse
import networkx as nx
import numpy as np

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges-file", required=True, help="File containing all edges")
    ap.add_argument("--embeddings-file", required=True, help="File containing all embeddings")

    args = vars(ap.parse_args())
    train_edges_file = args["edges_file"]
    embeddings_file = args["embeddings_file"]

    edges_path = train_edges_file.split("/")
    edges_filename = edges_path[-1].split(".")[0]
    spdist_file = "/".join(edges_path[:-1]) + "/" + edges_filename + "_spdist.npy"

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
    
    # Compute shortest path distances
    numnodes = len(dag.nodes())
    spdists = -1 * np.ones((numnodes, numnodes), dtype=np.float)
    lengths = dict(nx.algorithms.shortest_paths.unweighted.all_pairs_shortest_path_length(dag))
    for u in dag.nodes():
	for v, dist in lengths[u].iteritems():
	    spdists[u][v] = int(dist)
    spdists[spdists==-1] = int(spdists.max())

    np.save(spdist_file, spdists)
    print("Wrote shortest-path distances to:", spdist_file)
