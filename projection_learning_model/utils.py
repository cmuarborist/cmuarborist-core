# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function

import argparse
import os
import sys

import numpy as np

def compute_mrr(relevance, r=10):
    """Compute the mean reciprocal rank of a set of queries.

    relevance is a numpy matrix; each row contains the "relevance"
    of the predictions (= 0 or 1) made for each query.

    predictions are ranked in decreasing order of relevence.
    relevance[:, :15] are the top 15 most relevent predictions.

    The first non-zero entry of each row is the lowest ranked correct
    prediction. The reciprocal of this rank is the reciprocal rank.
    The mean of the reciprocal rank over all queries is returned as
    a percentage.

    Example:

        relevance = [[0, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 0, 0, 0]]

        ranks = [[2], [1], []]  # this is 0-based

        mrrs = [1/(2+1), 1/(1+1), 0] = [1/3, 1/2, 0]
    """
    ranks = [np.nonzero(t)[0] for t in relevance[:, :r]]
    mrrs = [1.0/(rank[0] + 1) if len(rank) > 0 else 0.0
            for rank in ranks]
    return 100.0 * np.mean(mrrs)

def compute_recall(relevance, r=10):
    true_parent_in_top_r = np.any(relevance[:, :r], axis=1)
    return 100.0 * np.mean(true_parent_in_top_r)

def compute_map(relevance, num_target_parents):
    """Compute the mean average precision of a set of queries.

    relevance is a numpy matrix; each row contains the "relevance"
    of the predictions (= 0 or 1) made for each query.

    num_target_parents contains the number of true parents for
    the given query (= 1 for the anon taxonomy).

    The precision@k is first calculated for each query and k;
    these are then average to get the AP for each query. The
    mean of the APs over all queries are then returned as a
    percentage.
    """
    # compute the cumulative number of correct predictions at each k
    cum_relevance = np.cumsum(relevance, axis=1)

    # construct the divisor to compute the AP; the divisor is = k
    # if the number of parents are greater than k, otherwise is =
    # the number of parents (to compare to SemEval competing methods)
    num_target_parents = num_target_parents.reshape(-1, 1)
    num_target_parents = num_target_parents.repeat(cum_relevance.shape[1],
                                                   axis=1)
    divisor = np.minimum(np.arange(1.0, cum_relevance.shape[1] + 1.0),
                         num_target_parents)

    # compute the precision at k for each k, by dividing the number
    # of correct predictions at each k by min(k, no. of parents)
    precision_at_k = cum_relevance / divisor

    aps = np.mean(precision_at_k, axis=1)
    return 100.0 * np.mean(aps)
