# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function

import os

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

from .constants import EPS
from .constants import NUM_THREADS
from .constants import NUM_WORKERS
from .constants import SAVE_EVERY
from .constants import SEED
from .constants import TRAIN_FRACTION
from .constants import VALIDATE_EVERY
from .constants import WEIGHT_NETWORK_LAYERS
from .constants import WEIGHT_NETWORK_HSIZE 
from .hyperparams import INIT_PROJECTION_STD
from .utils import compute_recall
from .utils import compute_mrr

if SEED is not None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
torch.set_num_threads(NUM_THREADS)


class NeuralTaxonomyExpander(nn.Module):

    def __init__(self, embeddings, num_projections, num_negative_samples=0,
                 child_counts=None, spdists=None, dropout_prob=0.0,
                 hard_negative_additional_prob=0.1, num_hidden_layers=WEIGHT_NETWORK_LAYERS,
                 hidden_layer_size=WEIGHT_NETWORK_HSIZE, loss_type="margin"):
        super(NeuralTaxonomyExpander, self).__init__()

        self.loss_type = loss_type

        vocab_size, embedding_size = embeddings.shape  # embedding_size = d
        embeddings = torch.FloatTensor(embeddings)

        # initialize word embeddings from the provided embeddings
        # use separate matrices for query (hyponym) and hypernym embeddings
        self.embedding_q = nn.Embedding.from_pretrained(embeddings, freeze=True)
        self.embedding_h = nn.Embedding.from_pretrained(embeddings, freeze=False)

        # initialize projector tensor (d x d x k) as identity + noise;
        # this makes the initial k projections = randomly corrupted
        # copies of the query embedding
        I = torch.eye(embedding_size, embedding_size)  # d x d
        I = I.unsqueeze(dim=0).repeat(num_projections, 1, 1)  # k x d x d
        O = torch.zeros(embedding_size, embedding_size)
        O = O.unsqueeze(dim=0).repeat(num_projections, 1, 1)
        noise = torch.normal(mean=O, std=INIT_PROJECTION_STD)
        self.projector = nn.Parameter(I + noise)

        # tanh neural network to generate weights from word/node embeddings
        weight_generator_network = []
        if num_hidden_layers > 0:
            weight_generator_network.extend([torch.nn.Linear(embedding_size,
                                                             hidden_layer_size),
                                             torch.nn.Tanh(),
                                             torch.nn.Dropout(dropout_prob)])  # input layer
            if num_hidden_layers > 1:
                for h in range(num_hidden_layers-1):  # hidden layers
                    weight_generator_network.extend([torch.nn.Linear(hidden_layer_size,
                                                                     hidden_layer_size),
                                                     torch.nn.Tanh(),
                                                     torch.nn.Dropout(dropout_prob)])
            weight_generator_network.append(torch.nn.Linear(hidden_layer_size,
                                                            num_projections))  # output layer

            self.b = torch.nn.Sequential(*[torch.nn.Linear(embedding_size, embedding_size),
                                           torch.nn.Tanh(),
                                           torch.nn.Dropout(dropout_prob),
                                           torch.nn.Linear(embedding_size, embedding_size)])
        else:
            weight_generator_network.append(torch.nn.Linear(embedding_size,
                                                            num_projections))
            self.b = torch.nn.Linear(embedding_size, embedding_size)
        self.W = torch.nn.Sequential(*weight_generator_network)

        # initialize dropouts
        self.dropout = nn.Dropout(dropout_prob)

        self.nnegs = num_negative_samples  # m
        self.hard_negative_additional_prob = hard_negative_additional_prob

        # distribution for negative sampling
        if child_counts is None:
            self.child_count_distribution = torch.FloatTensor(np.ones(vocab_size) /
                                                              float(vocab_size))
            self.internal_nodes = torch.LongTensor(np.arange(vocab_size))
        else:
            assert len(child_counts) == vocab_size
            # child_count_distribution = np.power(child_counts, 1.0)
            # child_count_distribution /= np.sum(child_count_distribution)
            # self.child_count_distribution = torch.FloatTensor(child_count_distribution)
            self.child_count_distribution = torch.FloatTensor(np.ones(vocab_size) /
                                                              float(vocab_size))
            self.internal_nodes = torch.LongTensor(np.nonzero(child_counts > 1)[0])

        # shortest-path distances for margin
        self.spdists = spdists
        if self.spdists is not None:
            self.spdists = torch.FloatTensor(spdists)  # vocab_size x vocab_size
            self.spdists /= self.spdists.max()  # normalize to be between 0.0 and 1.0

    def get_projection_matrix(self, hypernym):
        batch_size = hypernym.size(0)
        embedding_size = self.embedding_h.weight.size(1)
        num_projections = self.projector.size(0)
 
        W_h = self.W(self.embedding_h(hypernym)).view(batch_size, num_projections)  # B x k
        b_h = self.b(self.embedding_h(hypernym)).view(batch_size, 1, embedding_size)  # B x 1 x d

        projector_k_dd = self.projector.view(num_projections,
                                             embedding_size * embedding_size)  # k x d^2
        projector = torch.matmul(W_h, projector_k_dd).view(batch_size, embedding_size,
                                                           embedding_size)  # B x d x d

        return projector, b_h  # B x d x d, B x 1 x d

    def loss(self, query, hypernym):
        batch_size = query.size(0)  # B

        # get the hypernym embeddings for the batch: e_q
        query_embedding = self.embedding_q(query)
        query_embedding = self.dropout(query_embedding)
        embedding_size = query_embedding.size(1)

        # get the projection matrices and bias for the batch
        projector, bias = self.get_projection_matrix(hypernym)  # B x d x d, B x 1 x d

        # compute k-projections for the batch
        projection = torch.bmm(query_embedding.unsqueeze(dim=1),
                               projector)  + bias # B x 1 x d
        projection = self.dropout(projection)

        # get the hypernym embeddings for the batch: e_h
        hypernym_embedding = self.embedding_h(hypernym)  # B x d
        hypernym_embedding = self.dropout(hypernym_embedding)
        hypernym_embedding = hypernym_embedding.unsqueeze(1).permute(0, 2, 1)  # B x d x 1

        # compute s(u, v) = e_u M_v . e_v, size; B
        pos_similarity = torch.bmm(projection, hypernym_embedding).squeeze(dim=1)  # B x 1

        if self.nnegs > 0:
            neg_distributions = self.neg_distributions[query]
            
            if self.loss_type == "margin":
                # upweight hard negatives to help large-margin convegence
                with torch.no_grad():
                    node_dists = torch.mm(query_embedding.squeeze(),  # B x d, d x N
                                          self.embedding_h.weight.data.t())  # B x N
                    violations_mask = (node_dists - pos_similarity) > 0
                    valid_negs_mask = neg_distributions > 0
                    hard_negs_mask = violations_mask * valid_negs_mask
                    neg_distributions[hard_negs_mask] += self.hard_negative_additional_prob
                    neg_distributions /= neg_distributions.sum(dim=1, keepdim=True)

            # sample negative hypernyms for the batch
            neg_hypernyms = torch.multinomial(neg_distributions, self.nnegs,
                                              replacement=False)  # B x m
            
            # Loss = \sum_v' [s(u,v') - s(u,v) + gamma(v, v')]_+

            # get the hypernym embeddings of the negative hypernyms: e_h'
            neg_embedding = self.embedding_h(neg_hypernyms)  # B x m x d

            # get the negative projection matrices and bias
            unique_neg_hypernyms = neg_hypernyms.unique()
            neg_projector, neg_bias = self.get_projection_matrix(neg_hypernyms.view(-1, 1))
            neg_projector = neg_projector.view(batch_size, self.nnegs,
                                               embedding_size, embedding_size)  # B x m x d x d
            neg_bias = neg_bias.view(batch_size, self.nnegs, 1, embedding_size)  # B x m x 1 x d

            # compute k-projections for the batch: setup as BMM for efficiency
            query_Bm_1_d = query_embedding.repeat(1, self.nnegs).view(-1, embedding_size)
            query_Bm_1_d = query_Bm_1_d.unsqueeze(dim=1)  # Bm x 1 x d
            neg_projector_Bm_d_d = neg_projector.view(batch_size * self.nnegs,
                                                      embedding_size, embedding_size)
            neg_projection = torch.bmm(query_Bm_1_d, neg_projector_Bm_d_d)  # Bm x 1 x d
            neg_projection = neg_projection.view(batch_size, self.nnegs, 1,
                                                 embedding_size)  # B x m x 1 x d

            del neg_projector
            neg_projection = neg_projection + neg_bias
            neg_projection = neg_projection.squeeze()  # B x m x d
            neg_projection = self.dropout(neg_projection)
            neg_projection = neg_projection.view(batch_size, self.nnegs, embedding_size)

            # compute s(u, v') for v' (m negative samples), size: B x m
            neg_similarity = (neg_projection * neg_embedding).sum(dim=2)  # B x m

            if self.loss_type == "margin":
                # compute gamma(v, v'), size: B x m
                with torch.no_grad():
                    margin = self.spdists[neg_hypernyms.view(-1,1),
                                          hypernym.repeat(1, self.nnegs).view(-1,1)]  # Bm
                    margin = margin.view(batch_size, self.nnegs)  # B x m

            if self.loss_type == "margin":
                # scale s(u,v) with log-sigmoid to restrict range
                loss = (-pos_similarity.sigmoid().clamp(min=EPS) +
                        neg_similarity.sigmoid().clamp(min=EPS) +
                        margin.clamp(min=EPS)).clamp(min=0)  # B x m
                #print("Active triplets: " + str(loss.nonzero().size(0)) + " of " + str(loss.numel()))
            elif self.loss_type =="sgns":
                loss = (-pos_similarity.sigmoid().clamp(min=EPS).log().sum() -
                        (1.0 - neg_similarity.sigmoid()).clamp(min=EPS).log().sum())
        else:
            # Loss = -s(u, v)
            loss = -pos_similarity.sigmoid().clamp(min=EPS).log()

        return loss.sum()

    def fit(self, pairs, hypernym_sets=None, batch_size=2048, nepochs=100, learning_rate=0.01,
            save_folder="trained_model", idx2w=None, grad_clip=np.inf, weight_decay=0.0,
            start_epoch=0, ancestor_edges=None, npretrain=0):

        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

        if os.path.exists(save_folder + "/train_queries.txt") and\
            os.path.exists(save_folder + "/test_queries.txt"):
            train_queries = set(np.loadtxt(save_folder + "/train_queries.txt",
                                           delimiter=","))
            val_queries = set(np.loadtxt(save_folder + "/val_queries.txt",
                                         delimiter=","))
        else:
            # construct training and validation splits
            unique_queries = np.unique(pairs[:, 0])
            num_unique_queries = len(unique_queries)
            num_train_queries = int(TRAIN_FRACTION * num_unique_queries)
            np.random.shuffle(unique_queries)
            train_queries = set(unique_queries[:num_train_queries])
            val_queries = set(unique_queries[num_train_queries:])
            np.savetxt(save_folder + "/train_queries.txt",
                       unique_queries[:num_train_queries],
                       fmt="%d", delimiter=",")
            np.savetxt(save_folder + "/val_queries.txt",
                       unique_queries[num_train_queries:],
                       fmt="%d", delimiter=",")

        train_pairs = torch.LongTensor([p for p in pairs
                                        if p[0] in train_queries])
        validation_pairs = torch.LongTensor([p for p in pairs
                                             if p[0] in val_queries])

        # construct distribution for negative sampling
        num_rows = self.embedding_q.weight.size(0)
        neg_distributions = self.child_count_distribution.repeat(num_rows, 1)
        rows, cols = zip(*train_pairs)
        neg_distributions[rows, cols] = 0.0
        neg_distributions /= neg_distributions.sum(dim=1, keepdim=True)
        self.neg_distributions = neg_distributions

        if ancestor_edges is not None:
            # delete ancestors from negative samples
            rows, cols = zip(*ancestor_edges)
            neg_distributions[rows, cols] = 0.0
            neg_distributions /= neg_distributions.sum(dim=1, keepdim=True)
            self.neg_distributions = neg_distributions

        train_queries = list(train_queries)
        val_queries = list(val_queries)
        val_embeddings = self.embedding_q.weight[val_queries, :]
        num_training_pairs = len(train_pairs)
        num_validation_pairs = len(validation_pairs)
        train_dataloader = DataLoader(train_pairs, batch_size, shuffle=True,
                                      drop_last=False, num_workers=NUM_WORKERS)
        val_dataloader = DataLoader(validation_pairs, batch_size=batch_size, shuffle=True,
                                    drop_last=False, num_workers=0)

        train_loss_file = open(save_folder + "/train_loss.txt", "w")
        val_loss_file = open(save_folder + "/val_loss.txt", "w")

        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        npretrain = npretrain
        fn = lambda epoch: 1.0 if epoch < npretrain else 1.0
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, fn)

        nnegs_original = self.nnegs
        for epoch in tqdm(range(start_epoch, start_epoch + nepochs + npretrain), desc="epoch"):
            if epoch < npretrain:
                self.nnegs = 0
            else:
                self.nnegs = nnegs_original

            # training loss
            total_epoch_loss = 0.0
            total_pos_loss = 0.0
            total_neg_loss = 0.0
            self.train()
            for batch in train_dataloader:
                query = batch[:, 0]
                hypernym = batch[:, 1]

                loss = self.loss(query, hypernym)

                optimizer.zero_grad()  # zero old gradients
                loss.backward()  # compute new gradients
                clip_grad_norm_(self.parameters(), grad_clip)  # clip gradients
                optimizer.step()  # step in gradient direction

                total_epoch_loss += loss
            total_epoch_loss /= float(num_training_pairs * max(1, self.nnegs))
            scheduler.step()
            
            #print("epoch " + str(epoch+1) + ":", end=' ')
            #print("training loss=" + "{:.4f}".format(total_epoch_loss))
            train_loss_file.write("{:.4f}".format(total_epoch_loss) + "," +
                                  "{:.4f}".format(total_pos_loss) + "," +
                                  "{:.4f}".format(total_neg_loss) + "\n")
            train_loss_file.flush()

            # validation loss and metrics
            if num_validation_pairs > 0 and (epoch + 1) % VALIDATE_EVERY == 0:
                total_val_loss = 0.0
                total_pos_loss = 0.0
                total_neg_loss = 0.0
                self.eval()

                with torch.no_grad():
                    for batch in val_dataloader:
                        query = batch[:, 0]
                        hypernym = batch[:, 1]
                        loss = self.loss(query, hypernym)

                        total_val_loss += loss
                total_val_loss /= float(num_validation_pairs * max(1, self.nnegs))

                # get projectors for internal nodes
                W_h = self.W(self.embedding_h(self.internal_nodes)).data
                val_projector = torch.matmul(W_h, self.projector.data.permute(1, 0, 2))
                val_projector = val_projector.permute(1, 0, 2) # N' x d x d
                val_bias = self.b(self.embedding_h(self.internal_nodes)).data.unsqueeze(dim=1)

                # compute k-projections for the validation queries
                val_projections = torch.matmul(val_embeddings, val_projector)  # N' x V x d
                val_projections = val_projections + val_bias

                # get dot-products with the hypernym embeddings
                hypernym_embedding = self.embedding_h.weight.data[self.internal_nodes]  # N' x d
                hypernym_embedding = hypernym_embedding.unsqueeze(1).permute(0, 2, 1)  # N' x d x 1
                internal_dot_products = torch.bmm(val_projections, hypernym_embedding)  # N' x V x 1
                internal_dot_products = internal_dot_products.squeeze().permute(1, 0)  # V x N'
                del val_projector, val_bias, val_projections

                # compute dot-products of validation queries with internal nodes
                dot_products = -1e12 * torch.ones(len(val_queries),
                                                   self.embedding_h.weight.data.size()[0])  # V x N
                dot_products[:, self.internal_nodes] = internal_dot_products

                # compute MRR, MAP for top 15 predictions
                target_hypernyms = [hypernym_sets[int(validation_query)]
                                    for validation_query in val_queries]
                num_target_hypernyms = np.array([len(t) for t in target_hypernyms])
                _, pred_hypernyms = torch.sort(dot_products, dim=1,
                                               descending=True)
                pred_hypernyms = pred_hypernyms.int().numpy()
                relevance = [[int(ph in target_hypernyms[idx])
                              for ph in pred_hypernyms[idx]]
                             for idx in range(len(pred_hypernyms))]
                relevance = np.array(relevance)

                mrr = compute_mrr(relevance, r=1000000)
                rec = compute_recall(relevance, r=15)

                #print("val epoch " + str(epoch+1) + ":", end=' ')
                #print("val. loss=" + "{:.4f}".format(total_val_loss), end=' ')
                #print("mrr=" + "{:.4f}".format(mrr), end=' ')
                #print("rec=" + "{:.4f}".format(rec))
                val_loss_file.write("{:.4f}".format(total_val_loss) + "," +
                                    "{:.4f}".format(mrr) + "," +
                                    "{:.4f}".format(rec) + "," +
                                    "{:.4f}".format(total_pos_loss) + "," +
                                    "{:.4f}".format(total_neg_loss) + "\n")
                val_loss_file.flush()

            if (epoch + 1) % SAVE_EVERY == 0:
                #print("Saving model...")
                torch.save(self.state_dict(), save_folder + "/" + str(epoch+1) + ".pytorch")

        train_loss_file.close()
        val_loss_file.close()
        torch.save(self.state_dict(), save_folder + "/model.pytorch")
