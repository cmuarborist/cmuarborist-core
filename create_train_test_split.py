# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function

import argparse
import numpy as np

from collections import defaultdict
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--train-edges-file", required=True)
    ap.add_argument("--test-edges-file", required=True)
    ap.add_argument("--test-fraction", required=True, type=float)

    args = ap.parse_args()
    dataset = args.dataset
    train_edges_file = args.train_edges_file
    test_edges_file = args.test_edges_file
    test_fraction = args.test_fraction 
    
    all_edges = []
    all_nodes = set([])
    child_count = defaultdict(int)
    with open(dataset, "r") as f:
        for line in f:
            child, parent = line.strip().split(",")
            all_edges.append((child, parent))
            all_nodes.add(child)
            all_nodes.add(parent)
            child_count[parent] += 1
    
    leaf_nodes = []
    for node in all_nodes:
        if child_count[node] == 0:
            leaf_nodes.append(node)
    train_nodes, test_nodes = train_test_split(leaf_nodes, test_size=test_fraction)

    for node in all_nodes:
        if child_count[node] > 0:
            train_nodes.append(node)

    train_edges = [e for e in all_edges if e[0] in train_nodes and e[1] in train_nodes]
    test_edges = [e for e in all_edges if e[0] in test_nodes or e[1] in test_nodes]
    assert len(all_edges) == len(train_edges) + len(test_edges)

    print("Edges (all, train, test):", str(len(all_edges)) + " " + str(len(train_edges)) + " " + str(len(test_edges)))
    print("Nodes (all, train, test):", str(len(all_nodes)) + " " + str(len(train_nodes)) + " " + str(len(test_nodes)))
    open(train_edges_file, "w").write("\n".join([",".join(e) for e in train_edges]))
    open(test_edges_file, "w").write("\n".join([",".join(e) for e in test_edges]))
