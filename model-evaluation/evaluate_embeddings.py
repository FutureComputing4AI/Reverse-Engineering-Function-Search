# imports
import sys
import os
import time

import numpy as np

from evaluation_configs import get_configs
from evaluation_modules import get_knn, normalize_labels, compute_mrr
from refuse.utils.datasets import AssemblageFunctionsDataset

from sys import path
path.insert(1, '../')

start = time.time()

# more configurations
configs = get_configs()

# load test data set if normalizing labels
if configs['how_normalize'] is not None:
    assert configs['how_normalize'] in ['source', 'type', 'all'], """how_normalize must be in
[None, 'source', 'type', 'all'"""
    assert configs['dataset'] == 'assemblage', "label normalization is only supported with the assemblage dataset"
    dataset = AssemblageFunctionsDataset(**configs['dataset_configs'])

# load embeddings
embeddings = np.memmap(configs['embeddings_file'], dtype='float32', mode='r+')
embeddings = np.reshape(embeddings, (-1, configs['embd_size']))
labels = np.memmap(configs['labels_file'], dtype='int64', mode='r+')

# build NN search index using faiss
nearest_neighbors, distances = get_knn(embeddings, configs['dist_fn'], configs['K'], configs['M'])

# normalize labels
if configs['how_normalize'] is not None:
    labels, _ = normalize_labels(labels, dataset, configs['how_normalize'])

# compute mrr and recall
mrr_upper, mrr_lower, recalls  = compute_mrr(labels, nearest_neighbors, ks=configs['recall_ks'])

end = time.time()

with open(configs['output_file'], 'a+') as out:
    print("Results for normalize = ", configs['how_normalize'], file=out)
    print("The lower and upper bounds on the mean reciprocal rank over the test set are: ", mrr_lower, mrr_upper, file=out)
    print("The recalls are: ", recalls, file=out)
    print("The time to evaluate was {} seconds".format(end-start), file=out)
    print(file=out)
