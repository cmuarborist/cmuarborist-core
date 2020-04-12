# -*- coding: utf-8 -*-

NUM_WORKERS = 0  # number of additional data-looading workers
NUM_THREADS = 24  # number of tensor-processing threads
SAVE_EVERY = 5  # model-saving interval in epochs
VALIDATE_EVERY = 1  # validation interval in epochs
SEED = None  # random seed for Torch
TRAIN_FRACTION = 0.85  # fraction of data used for training (not validation)
EPS = 1e-9  # small float value, used to prevent underflow issues
WEIGHT_NETWORK_LAYERS = 0 # number of hidden layers in the weight generator network
WEIGHT_NETWORK_HSIZE  = 150 # hidden layer size in the weight generator network
