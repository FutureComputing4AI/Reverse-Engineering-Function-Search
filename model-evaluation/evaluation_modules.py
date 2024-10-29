# imports
import sys
import os
import time

import numpy as np
from tqdm import tqdm
import faiss
import torch

# debug NANs
import jax
jax.config.update("jax_debug_nans", True)

def get_knn(embeddings, dist_fn, K=30, M=32):
    """
        Use FAISS HNSW index to compute (approximate) KNNs

        Params:
            embeddings:    numpy array, shape (N, E), of embeddings
            dist_fn:       which distance function to use to compare embeddings,
                           either 'euclid' or 'cosine'
            K:             number of nearest neighbors to calculate for each query point 
            M:             from the FAISS docs - the number of neighbors used in the graph.
                           A larger M is more accurate but uses more memory.

        Returns:
            neighbors:     numpy array, shape (N, K), where neighbors[i][j] = k means that
                           embeddings[k] is the j-th nearest neighbor of embeddings[i]
            distances:     numpy array, shape (N, K), where distances[i][j] is the distance
                           between i and its j-th nearest neighbor
                           
    """

    
    embd_size = embeddings.shape[-1]

    # choose metric
    if dist_fn=='euclid':
        metric = faiss.METRIC_L2
    elif dist_fn=='cosine':
        metric = faiss.METRIC_INNER_PRODUCT
        faiss.normalize_L2(embeddings) # normalization => inner product = cosine similarity
    else:
        raise ValueError("Unsupported distance function {}. Supported distance functions \
                          are 'euclid' and 'cosine.".format(dist_fn), file=sys.stderr)
    
    embeddings = torch.from_numpy(embeddings)

    start = time.time()
    index = faiss.IndexHNSWFlat(embd_size, M, metric)
    index.add(embeddings)
    distances, nearest_neighbors = index.search(embeddings, K+1)

    # nearest neighbor will always be itself, so remove this
    distances = distances[:,1:]
    nearest_neighbors = nearest_neighbors[:,1:] 
    end = time.time()

    print("The time to build the faiss KNN index is ", end-start, file=sys.stderr)
    sys.stdout.flush()

    return nearest_neighbors, distances

def mask_types(name, mask_char = '#'):

    """
        mask types in function names

        params:
            name:         original function name (string)
            mask_char:    which character/string to use in the mask, default '#'

        returns:
            masked_name:  masked function name (string)
    """
    
    masked_name = ''
    mask_on = [False]

    try:
        open_count = name.count('<')
        closed_count = name.count('>')
        operator_count = name.count('operator')
        
        for i, char in enumerate(name):
            if char == '<': # path 1
                masked_name += char 
                if (name[i+1:i+8] == 'lambda_'): # apparently out-of-range slicing is handled gracefully in python
                    mask_on.append(False) # path 2
                elif open_count > closed_count: # path 3
                    if ((name[i-8:i] == 'operator') or (name[i-9:i] == 'operator<')): # path 4
                        open_count -= 1
                        continue
                    else: # path 5
                        mask_on.append(True)
                        if name[i+1:i+2] != '<': # path 6
                            masked_name += mask_char # absent 'else' clause is path 20
                else: # path 7
                    mask_on.append(True)
                    if name[i+1:i+2] != '<': # path 8
                        masked_name += mask_char # absent 'else' clause is path 21

            elif char == '>': # path 9
                masked_name += char
                if closed_count - open_count > 0: # path 10
                    if ((name[i-8:i] == 'operator') 
                        or (name[i-9:i] == 'operator-')
                        or (name[i-9:i] == 'operator>')): # path 11
                        closed_count -= 1
                        continue
                    else: # path 12
                        mask_on.pop()
                        if ((name[i+1:i+2] not in ['<', '>', ',']) and (mask_on[-1])): # path 13
                            masked_name += mask_char # absent 'else' clause is path 22
                else: # path 14
                    mask_on.pop()
                    if ((name[i+1:i+2] not in ['<', '>', ',']) and (mask_on[-1])): # path 15
                        masked_name += mask_char # absent 'else' clause is path 23

            elif ((char == ',') and (mask_on[-1])): # path 16
                masked_name += char 
                if name[i+1:i+2] != '<': # avoid index out of range error
                    masked_name += mask_char # path 17, absent 'else' clause is path 24
            elif not mask_on[-1]: # path 18
                masked_name += char # absent 'else' clause is path 19

    except Exception as e: # path 25
        print("ERROR: {}".format(e), file=sys.stderr)
        print("Error occurred while masking name {}, reverting to original name".format(name), file=sys.stderr)
        print(file=sys.stderr)
        return name
    
    return masked_name

def normalize_fn_name(name, how):
    """
        normalize function names

        params:
            name:  function name
            how:   normalization method, either 'source', 'type', or 'all' (source and type)

        returns:
            normalized name

    """
    
    if how=='source':
        parts = name.split('\\')
        return parts[-1]
    elif how=='type':
        return mask_types(name)
    elif how=='all':
        parts = name.split('\\')
        return mask_types(parts[-1])
    else:
        raise ValueError("how={} is undefined".format(how))

def normalize_labels(labels, dataset, how, mapping={}):

    """
        normalize labels

        params:
            labels:  labels to normalize
            dataset: Dataset class with method get_name, that takes in an integer label
                     and retrieves the associated function name (string)
            how:     normalization strategy: one of 'source', 'type', or 'all'
            mapping: mapping of normalized function names to new integer labels
                     new entries are added with mapping[new_name] = len(mapping)
                     defaults to an empty dictionary

        returns:
            normalized_labels
            mapping
    """
    
    orig_shape = labels.shape
    labels = labels.flatten()
    
    new_labels = np.zeros_like(labels)

    for i, label in tqdm(enumerate(labels)):
        name = dataset.get_name(label)
        normalized_name = normalize_fn_name(name, how)
        if normalized_name in mapping:
            new_labels[i] = mapping[normalized_name]
        else:
            mapping[normalized_name] = len(mapping)
            new_labels[i] = mapping[normalized_name]

    new_labels = new_labels.reshape(orig_shape)

    return new_labels, mapping

def compute_mrr(labels, neighbors, ks=[1]):
    """
        compute mean reciprocal rank (upper and lower bounds) 

        params:
            labels:     numpy array of shape (N,)
            neighbors:  numpy array of shape (N, K) where K is the number of neighbors
            ks:         where to compute recall, should have type int, list, or None

        returns:
            upper:  MRR upper bound    
            lower:  MRR lower bound
            recalls: dictionary with keys k, values recall @k for k in ks

    """

    if type(ks) is int:
        ks = [ks]
    elif type(ks) is None:
        ks = []
    assert type(ks) is list, "ks must have type int, list, or None"
    for k in ks:
        assert ((type(k) is int) and (k > 0)), "k in ks must be a positive int"
    
    neighbors_labels = np.take(labels, neighbors)
    hits = (neighbors_labels == np.repeat(np.expand_dims(labels, -1), neighbors.shape[-1], axis=1))
    any_hits = np.any(hits, axis=1)

    # compute upper and lower bounds on reciprocal rank
    true_rr = 1./(np.argmax(hits, axis=1)+1) # argmax returns the index where the first True appears
    upper_bound = 1./(neighbors.shape[-1] + 1)
    lower_bound = 0

    upper = np.mean(np.where(any_hits, true_rr, upper_bound))
    lower = np.mean(np.where(any_hits, true_rr, lower_bound))

    # compute recall
    recalls = {}

    unique_labels, indices, counts = np.unique(labels, return_inverse = True, return_counts = True)
    total_num_hits = np.take(counts, indices)-1 #subtract off the point itself
    total_num_hits = np.where(total_num_hits==0, 1, total_num_hits) # make sure denominator is not zero

    for k in ks:
        hits_in_top_k = np.sum(hits[:, :k], axis=1)
        recalls[k] = (hits_in_top_k > 0).mean()

    return upper, lower, recalls
