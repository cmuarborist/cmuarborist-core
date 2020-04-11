# -*- coding: utf-8 -*-

NUM_WORKERS = 8  # number of additional data-looading workers
NUM_THREADS = 8  # number of tensor-processing threads
SAVE_EVERY = 1  # model-saving interval in epochs
VALIDATE_EVERY = 1  # validation interval in epochs
SEED = None  # random seed for Torch
TRAIN_FRACTION = 0.85  # fraction of data used for training (not validation)
EPS = 1e-9  # small float value, used to prevent underflow issues
