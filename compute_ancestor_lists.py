# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function

import argparse
from collections import defaultdict
import random
import sys
import networkx as nx

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges-file", required=True, help="File containing all edges")

    args = vars(ap.parse_args())
    edges_file = args["edges_file"]

    edges_path = edges_file.split("/")
    edges_filename = edges_path[-1].split(".")[0]
    ancestors_file = "/".join(edges_path[:-1]) + "/" + edges_filename + "_ancestors.csv"

    tree = defaultdict(list)
    nodes = set([])
    with open(edges_file, "r") as f:
        for line in f:
            child, parent = line.strip().split(",")
            nodes.add(child)
            nodes.add(parent)
            tree[parent].append(child)

    G = nx.from_dict_of_lists(tree, create_using=nx.DiGraph())

    with open(ancestors_file, "w") as f:
        for node in nodes:
            ancestors = nx.ancestors(G, node)
            if len(ancestors) > 0:
                f.write(node + "," + ",".join(ancestors) + "\n")

    print("Wrote ancestors to: " + ancestors_file)
