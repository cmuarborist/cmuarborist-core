# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function

import argparse
import networkx as nx
import nltk
import pandas as pd
import os

from networkx.algorithms.traversal.depth_first_search import dfs_tree
from nltk.corpus import wordnet as wn
from tqdm import tqdm

if __name__ == "__main__":
    nltk.download('wordnet')
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-folder", required=True)

    args = ap.parse_args()
    data_folder = args.data_folder

    try:
	os.mkdir(data_folder)
    except OSError as error:
        print("Ignoring error:", error.strerror)

    # Construct Wordnet Graph

    print("Constructing Wordnet graph...")
    edges = set()  # child, parent, type
    for synset in tqdm(wn.all_synsets(pos='n')):
	for hyper in synset.hypernyms():
	    edges.add((synset.name(), hyper.name(), "hypernym"))
	    
	for instance in synset.instance_hyponyms():
	    for hyper in instance.instance_hypernyms():
		edges.add((instance.name(), hyper.name(), "hypernym"))
		for h in hyper.hypernyms():
		    edges.add((instance.name(), h.name(), "hypernym"))
	
	for substance_holonym in synset.substance_holonyms():
	    edges.add((synset.name(), substance_holonym.name(), "substance_holonym"))
	
	for part_holonym in synset.part_holonyms():
	    edges.add((synset.name(), part_holonym.name(), "part_holonym"))

    nouns = pd.DataFrame(list(edges), columns=['child', 'parent', 'type'])

    # Get Animal Subgraph

    G = nx.DiGraph()
    for child, parent, edge_type in edges:
	G.add_edge(parent, child, edge_type=edge_type)

    subgraph = dfs_tree(G, source="animal.n.01")
    subgraph = G.subgraph(subgraph.nodes)

    print("Number of edges in subgraph:", len(subgraph.edges))
    print("Subgraph depth (longest path length): ", nx.dag_longest_path_length(subgraph))

    print("Edge types:")
    all_nodes = set(subgraph.nodes)
    mammals = nouns[(nouns["child"].isin(all_nodes) &
		    (nouns["parent"].isin(all_nodes)))]
    print(mammals.groupby("type").count())

    # Save to disk

    with open(data_folder + "/mammal.csv", "w") as f:
	for parent, child in subgraph.edges:
	    f.write(child.split(".")[0] + "," + parent.split(".")[0] + "\n")
    with open(data_folder + "/mammal_vocab.csv", "w") as f:
	for node in subgraph.nodes:
	    f.write(node.split(".")[0].replace("_", " ") + "\n")
