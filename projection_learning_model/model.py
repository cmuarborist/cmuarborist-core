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
from tqdm import tqdm as tqdm

from .constants import EPS
from .constants import NUM_THREADS
from .constants import NUM_WORKERS
from .constants import SAVE_EVERY
from .constants import SEED
from .constants import TRAIN_FRACTION
from .constants import VALIDATE_EVERY
from .hyperparams import INIT_PROJECTION_STD
from .utils import compute_recall
from .utils import compute_mrr

if SEED is not None:
    torch.manual_seed(SEED)
torch.set_num_threads(NUM_THREADS)


class NeuralTaxonomyExpander(nn.Module):

    def __init__(self, embeddings, num_projections, num_negative_samples=0,
                 child_counts=None, dropout_prob=0.0):
        super(NeuralTaxonomyExpander, self).__init__()

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

        # initialize affine transformation of hypernym projections
        self.W = nn.Parameter(torch.normal(mean=torch.zeros(1, num_projections),
                              std=INIT_PROJECTION_STD))
        self.b = nn.Parameter(torch.normal(mean=torch.zeros(1, embedding_size),
                              std=INIT_PROJECTION_STD))

        # initialize dropouts
        self.dropout = nn.Dropout(dropout_prob)

        self.nnegs = num_negative_samples  # m

        # distribution for negative sampling
        if child_counts is None:
            self.child_count_distribution = torch.FloatTensor(np.ones(vocab_size) /
                                                              float(vocab_size))
        else:
            assert len(child_counts) == vocab_size
            child_count_distribution = np.power(child_counts, 1.0)
            child_count_distribution /= np.sum(child_count_distribution)
            self.child_count_distribution = torch.FloatTensor(child_count_distribution)

    def forward(self, query_embedding):
        # compute k-projections for the batch
        projection = torch.matmul(query_embedding,
                                  self.projector).permute(1, 0, 2)  # B x k x d
        projection = self.dropout(projection)

        # compute final projections via linear combination
        projection = torch.matmul(self.W, projection) + self.b  # B x 1 x d

        return projection

    def loss(self, query, projection, hypernym):
        batch_size = projection.size(0)  # B

        # get the hypernym embeddings for the batch: e_h
        hypernym_embedding = self.embedding_h(hypernym)  # B x d
        hypernym_embedding = self.dropout(hypernym_embedding)
        hypernym_embedding = hypernym_embedding.unsqueeze(1).permute(0, 2, 1)  # B x d x 1

        # dot-products between the projection and hypernym embedding: P . e_h
        similarity = torch.bmm(projection, hypernym_embedding).squeeze(1)  # B x 1

        # compute the first component of the loss: log-sigmoid(P . e_h)
        # note: clamping the sigmoid is needed to avoid log(0) = -inf
        pos_loss = similarity.sigmoid().clamp(min=EPS).log().sum()

        neg_loss = 0.0
        if self.nnegs > 0:
            neg_distributions = self.neg_distributions[query]

            # sample negative hypernyms for the batch
            neg_hypernyms = torch.multinomial(neg_distributions, self.nnegs,
                                              replacement=True)  # B x m
            neg_hypernyms = neg_hypernyms.view(batch_size, self.nnegs)

            # get the hypernym embeddings of the negative hypernyms: e_h'
            neg_embedding = self.embedding_h(neg_hypernyms)  # B x m x d
            neg_embedding = neg_embedding.permute(0, 2, 1)  # B x d x m

            # dot-products between the projections and negative hypernym embeddings: P . e_h'
            neg_similarity = torch.bmm(projection, neg_embedding)  # B x 1 x m
            neg_similarity = neg_similarity.view(batch_size * self.nnegs)  # Bm

            # compute the second component of the loss log-sigmoid(P . e_h')
            neg_loss = (1.0 - neg_similarity.sigmoid()).clamp(min=EPS).log().sum()

        return -pos_loss, -neg_loss

    def fit(self, pairs, hypernym_sets=None, batch_size=2048, nepochs=100, learning_rate=0.01,
            save_folder="trained_model", idx2w=None, grad_clip=np.inf, weight_decay=0.0,
            start_epoch=0, ancestor_edges=None):

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
                                    drop_last=False, num_workers=NUM_WORKERS)

        train_loss_file = open(save_folder + "/train_loss.txt", "w")
        val_loss_file = open(save_folder + "/val_loss.txt", "w")

        optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        for epoch in tqdm(range(start_epoch, start_epoch + nepochs), desc="epoch"):

            # training loss
            total_epoch_loss = 0.0
            total_pos_loss = 0.0
            total_neg_loss = 0.0
            self.train()
            for batch in train_dataloader:
                query = batch[:, 0]
                hypernym = batch[:, 1]

                query_embedding = self.embedding_q(query)  # get query embeddings
                query_embedding = self.dropout(query_embedding)
                projection = self.forward(query_embedding)  # compute projections
                pos_loss, neg_loss = self.loss(query, projection, hypernym)
                loss = pos_loss + neg_loss

                optimizer.zero_grad()  # zero old gradients
                loss.backward()  # compute new gradients
                clip_grad_norm_(self.parameters(), grad_clip)  # clip gradients
                optimizer.step()  # step in gradient direction

                total_epoch_loss += loss
                total_pos_loss += pos_loss
                total_neg_loss += neg_loss
            total_epoch_loss /= float(num_training_pairs)
            total_pos_loss /= float(num_training_pairs)
            total_neg_loss /= float(num_training_pairs)

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

                for batch in val_dataloader:
                    query = batch[:, 0]
                    hypernym = batch[:, 1]

                    query_embedding = self.embedding_q(query)  # get query embeddings
                    projection = self.forward(query_embedding)  # compute projections
                    pos_loss, neg_loss = self.loss(query, projection, hypernym)
                    loss = pos_loss + neg_loss
                    total_val_loss += loss
                    total_pos_loss += pos_loss
                    total_neg_loss += neg_loss
                total_val_loss /= float(num_validation_pairs)
                total_pos_loss /= float(num_validation_pairs)
                total_neg_loss /= float(num_validation_pairs)

                # compute MRR, MAP for top 15 predictions
                target_hypernyms = [hypernym_sets[int(validation_query)]
                                    for validation_query in val_queries]
                num_target_hypernyms = np.array([len(t) for t in target_hypernyms])
                projections = self.forward(val_embeddings).squeeze()

                dot_products = torch.mm(projections,
                                        self.embedding_h.weight.t())
                _, pred_hypernyms = torch.sort(dot_products, dim=1,
                                               descending=True)
                pred_hypernyms = pred_hypernyms.int().numpy()
                relevance = [[int(ph in target_hypernyms[idx])
                              for ph in pred_hypernyms[idx]]
                             for idx in range(len(pred_hypernyms))]
                relevance = np.array(relevance)

                mrr = compute_mrr(relevance, r=100000)
                mean_ap = compute_recall(relevance, r=15)

                #print("val epoch " + str(epoch+1) + ":", end=' ')
                #print("val. loss=" + "{:.4f}".format(total_val_loss), end=' ')
                #print("mrr=" + "{:.4f}".format(mrr), end=' ')
                #print("rec=" + "{:.4f}".format(mean_ap))
                val_loss_file.write("{:.4f}".format(total_val_loss) + "," +
                                    "{:.4f}".format(mrr) + "," +
                                    "{:.4f}".format(mean_ap) + "," +
                                    "{:.4f}".format(total_pos_loss) + "," +
                                    "{:.4f}".format(total_neg_loss) + "\n")
                val_loss_file.flush()

            if (epoch + 1) % SAVE_EVERY == 0:
                #print("Saving model...")
                torch.save(self.state_dict(), save_folder + "/" + str(epoch+1) + ".pytorch")

        train_loss_file.close()
        val_loss_file.close()
        torch.save(self.state_dict(), save_folder + "/model.pytorch")
