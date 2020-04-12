# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function

import argparse
import networkx as nx
import numpy as np
import large_margin_taxonomic_role_model.model as projection_learning_model
import torch

from large_margin_taxonomic_role_model.constants import SAVE_EVERY
from large_margin_taxonomic_role_model.utils import compute_mrr, compute_recall

if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--nprojections", required=True)
    ap.add_argument("--embeddings-file", required=True)
    ap.add_argument("--train-edges-file", required=True)
    ap.add_argument("--test-edges-file", required=True)
    ap.add_argument("--model-file", required=False)
    ap.add_argument("--save-folder", required=True,
                    help="Folder to store the trained model and losses")

    args = ap.parse_args()
    num_projections = int(args.nprojections)
    embeddings_file = args.embeddings_file
    train_edges_file = args.train_edges_file
    test_edges_file = args.test_edges_file
    save_folder = args.save_folder
    model_file = args.model_file

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

    # Load train and test edges
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

    # load test edges
    test_edges = []  # row indices of child-parent pairs
    test_parent_sets = {}  # compatability for multiple parents

    with open(test_edges_file, "r") as f:
	for line in f:
	    child_topicid, parent_topicid = line.strip().split(",")
	    child_idx, parent_idx = topicid2row[child_topicid], topicid2row[parent_topicid]
	    test_edges.append((child_idx, parent_idx))
	    test_parent_sets[child_idx] = set([parent_idx])
	    dag.add_edge(child_idx, parent_idx)
    test_edges = np.array(test_edges)

    # compute child counts
    child_counts = np.zeros(len(embeddings), dtype=np.float)
    for child, parent in train_edges:
        child_counts[parent] += 1
    assert int(child_counts.sum()) == len(train_edges)
    child_counts += 1.
    internal_nodes = np.nonzero(child_counts>1)[0]

    if model_file:
        saved_model = torch.load(model_file)
    else:
        # pick model with best validation MRR
        val_loss = np.loadtxt(save_folder + "/val_loss.txt", delimiter=",")
        best_epoch = val_loss[:, 1].argmax()
        best_epoch = int(SAVE_EVERY * round(float(best_epoch)/SAVE_EVERY))
        print("Best epoch (highest MRR): " + str(best_epoch))
        saved_model = torch.load(save_folder + "/"  + str(best_epoch) + ".pytorch")

    nte = projection_learning_model.NeuralTaxonomyExpander(embeddings=embeddings, num_projections=num_projections)
    nte.load_state_dict(saved_model)
    nte.eval()  # put model in evaluation mode

    # compute metrics on test data
    test_queries = np.array(test_parent_sets.keys())
    test_embeddings = torch.FloatTensor(embeddings[test_queries, :])
    target_parents = [test_parent_sets[int(q)] for q in test_queries]

    W_h = nte.W(nte.embedding_h(nte.internal_nodes)).data
    test_projector = torch.matmul(W_h, nte.projector.data.permute(1, 0, 2))
    test_projector = test_projector.permute(1, 0, 2) # N' x d x d
    test_bias = nte.b(nte.embedding_h(nte.internal_nodes)).data.unsqueeze(dim=1)

    # compute k-projections for the validation queries
    test_projections = torch.matmul(test_embeddings, test_projector)  # N' x V x d
    test_projections = test_projections + test_bias

    # get dot-products with the hypernym embeddings
    hypernym_embedding = nte.embedding_h.weight.data[nte.internal_nodes]  # N' x d
    hypernym_embedding = hypernym_embedding.unsqueeze(1).permute(0, 2, 1)  # N' x d x 1
    internal_dot_products = torch.bmm(test_projections, hypernym_embedding)  # N' x V x 1
    internal_dot_products = internal_dot_products.squeeze().permute(1, 0)  # V x N'
    del test_projector, test_bias, test_projections

    # compute dot-products of validation queries with internal nodes
    dot_products = -1e12 * torch.ones(len(test_queries),
                                               nte.embedding_h.weight.data.size()[0])  # V x N
    dot_products[:, nte.internal_nodes] = internal_dot_products

    dot_products[torch.LongTensor(range(len(dot_products))), test_queries] = -1e12
    parent_dists, pred_parents = torch.sort(dot_products, dim=1, descending=True)
    pred_parents = pred_parents.int().numpy()
    relevance = [[int(ph in target_parents[idx])
		  for ph in pred_parents[idx]]
		 for idx in range(len(pred_parents))]
    relevance = np.array(relevance)
    mrr = compute_mrr(relevance, r=pred_parents.shape[1])
    recall = compute_recall(relevance, r=15)

    shortest_path_distances = []
    for test_idx in range(len(pred_parents)):
	for true_parent_idx in target_parents[test_idx]:
	    try:
		shortest_path_distances.append(nx.shortest_path_length(dag,
								       pred_parents[test_idx][0],
								       true_parent_idx))
	    except nx.NetworkXNoPath:
		shortest_path_distances.append(-1)
    shortest_path_distances = np.array(shortest_path_distances)
    shortest_path_distances[shortest_path_distances==-1] = 18
    mean_spl = np.mean(shortest_path_distances)

    print("MRR = {:.3f}%, Recall@15 = {:.3f}%, d(v, vhat(u)) = {:.3f}".format(mrr, recall, mean_spl))
