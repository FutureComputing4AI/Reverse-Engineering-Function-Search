import sys
import itertools
import collections

import random
import numpy as np
import torch
from torch.utils.data.sampler import Sampler

import jax
from jax import numpy as jnp
from jax import vmap, lax

def build_triplet_mask(embeddings, labels):
    '''
        makes a mask of valid triplets
        for each valid triplet (x, y, z), (embeddings[x], embeddings[y], embeddings[z]) must be distinct
            from the embeddings of any other valid triplet

        inputs:  embeddings array (batch_size, embedding_size)
                 labels array (batch_size,)
                 
        returns: mask (batch_size, batch_size, batch_size) where
                 mask[x][y][z] = 1 if  labels[x] = labels[y], labels[x] != labels[z]
                                   AND x != y
                                   AND embeddings[x] != embeddings[y]
                                   AND there is no other (a, b, c) s.t.
                                       mask[a][b][c] = 1,
                                       (embeddings[a], embeddings[b], embeddings[c]) = (
                                            embeddings[x], embeddings[y], embeddings[z])
                                 0 OTHERWISE      
    '''

    labels1 = jnp.expand_dims(labels, 1)
    labels2 = jnp.expand_dims(labels, 0)
    pos_mask = jnp.asarray((labels1==labels2), dtype=jnp.uint8)
    neg_mask = pos_mask ^ 1

    diag = jnp.identity(pos_mask.shape[0])
    pos_mask = pos_mask - diag

    # valid_triplets[i][j][k] tells if (i,j,k) a valid anchor, positive, negative according to label[i], label[j], label[k]
    valid_triplets_mask = jnp.expand_dims(pos_mask, 2) * jnp.expand_dims(neg_mask, 1)
    
    # remove duplicates
    _, unique_indices = jnp.unique(embeddings, return_index=True, axis=0, size=embeddings.shape[0])
    unique_indices_mask = jnp.isin(jnp.arange(embeddings.shape[0]), unique_indices)

    tmp = jnp.expand_dims(unique_indices_mask, 0) * jnp.expand_dims(unique_indices_mask, 1)
    unique_triplets_mask = jnp.expand_dims(tmp, 2) * jnp.expand_dims(tmp, 1)

    triplets_mask = valid_triplets_mask * unique_triplets_mask    
        
    return triplets_mask

def loss_semihard_mining(embeddings, labels, margin, dist_fn):
    '''
        returns the semihard loss for a mini-batch of triplets

        each anchor is used in one triplet only

        negatives:
        the hardest semi-hard negative is selected
        if there are no semihard negatives, the easiest hard negative is selected
        if all the negatives are easy, for every positive of that anchor, then that anchor does not contribute to the loss

        inputs:  embeddings array (batch_size, embeddings size)
                 labels array, (batch_size,)
                 margin: nonnegative number
                 dist_fn: all pairs distance function
                 
        returns: loss from that batch

    '''

    assert margin >= 0, "margin must be nonnegative"
    epsilon = 1e-5

    #1. get a mask of valid triplets
    valid_triplets = build_triplet_mask(embeddings, labels)

    #2. compute the distance matrix
    dist_matrix = dist_fn(embeddings, embeddings)

    #3. compute the triplet margin
    #   triplet_margin[a][p][n] = d(a, n) - d(a, p)
    triplet_margin = jnp.expand_dims(dist_matrix, 1) - jnp.expand_dims(dist_matrix, 2) 

    #4. set semihard, hard threshold conditions
    threshold_semihard = ((triplet_margin <= margin) & (triplet_margin > 0))
    threshold_hard = (triplet_margin <= 0)

    #5. get the hardest semihard triplet for each anchor
    losses = (-1 * triplet_margin) + margin
    valid_semihard_triplets = losses * threshold_semihard * valid_triplets
    #max over possible semihard negatives, then over possible positives
    best_semihard_triplets = jnp.max(jnp.max(valid_semihard_triplets, axis=-1), axis=-1) 

    #6. get the easiest hard triplet for each anchor
    valid_hard_triplets = losses * threshold_hard * valid_triplets
    # when taking the minimum, use, infinity, instead of 0, as the mask value
    valid_hard_triplets_inf = jnp.where(valid_hard_triplets==0, jnp.inf, valid_hard_triplets)
    #min over possible hard negatives, then over possible positives
    best_hard_triplets_inf = jnp.min(jnp.min(valid_hard_triplets_inf, axis=-1), axis=-1) 
    best_hard_triplets = jnp.where(best_hard_triplets_inf==jnp.inf, 0, best_hard_triplets_inf)

    #7. for each anchor, pick the hardest semihard triplet, or the easiest hard triplet when there are no semihard triplets
    losses_best_triplets = jnp.where(best_semihard_triplets==0, best_hard_triplets, best_semihard_triplets)
    denom = jnp.max(jnp.array([jnp.count_nonzero(losses_best_triplets), epsilon])) # avoid divide by zero error
    loss = jnp.sum(losses_best_triplets) / denom

    return loss

def get_labels_to_indices_fork(labels):
    """
    Creates labels_to_indices, which is a dictionary mapping each label
    to a list of indices that will be used to index into self.dataset

    This is a helper function for MPerClassSamplerFork, and is forked from
    pytorch-metric-learning to have values in the dictionary be lists 
    instead of numpy arrays.
    """

    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    labels_to_indices = collections.defaultdict(list)
    for i, label in enumerate(labels):
        labels_to_indices[label].append(i)
    return labels_to_indices

class MPerClassSamplerFork(Sampler):
    """
    At every iteration, this will return m samples per class. For example,
    if dataloader's batchsize is 100, and m = 5, then 20 classes with 5 samples
    each will be returned

    Forked from pytorch-metric-learning, replaces np.random.shuffle with faster alternatives
    uses a combination of np.random.default_rng and python's built in random functions
    optimized for M = 2
    """

    def __init__(self, labels, m, batch_size=None, length_before_new_iter=100000, seed=70):
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
            
        self.rng = np.random.default_rng(seed=seed)
        random.seed(seed)
        
        self.m_per_class = int(m)
        self.batch_size = int(batch_size) if batch_size is not None else batch_size
        self.labels_to_indices = get_labels_to_indices_fork(labels)
        self.labels = np.array(list(self.labels_to_indices.keys()))
        self.length_of_single_pass = self.m_per_class * len(self.labels)
        self.list_size = length_before_new_iter
        if self.batch_size is None:
            if self.length_of_single_pass < self.list_size:
                self.list_size -= (self.list_size) % (self.length_of_single_pass)
        else:
            assert self.list_size >= self.batch_size
            assert (
                self.length_of_single_pass >= self.batch_size
            ), "m * (number of unique labels) must be >= batch_size"
            assert (
                self.batch_size % self.m_per_class
            ) == 0, "m_per_class must divide batch_size without any remainder"
            self.list_size -= self.list_size % self.batch_size

    def __len__(self):
        return self.list_size

    def __iter__(self):
        idx_list = [0] * self.list_size
        j = 0
        num_iters = self.calculate_num_iters()
        for _ in range(num_iters):
            if self.batch_size is None:
                curr_label_set = self.labels
            else:
                curr_label_set = self.rng.choice(self.labels, size=((self.batch_size // self.m_per_class),), 
                                                 replace=False, shuffle=False).tolist()
            
            batch = list(map(self.get_sample_for_label, curr_label_set))
            batch = list(itertools.chain.from_iterable(batch))
            local_batch_size = len(batch)
            idx_list[j:j+local_batch_size] = batch
            j += local_batch_size
            
        return iter(idx_list)

    def calculate_num_iters(self):
        divisor = (
            self.length_of_single_pass if self.batch_size is None else self.batch_size
        )
        return self.list_size // divisor if divisor < self.list_size else 1
    
    def get_sample_for_label(self, label):
        t = self.labels_to_indices[label]
        replace = (len(t) < self.m_per_class)
        if replace:
            return random.choices(t, k=self.m_per_class)
        else:
            return random.sample(t, k=self.m_per_class)
