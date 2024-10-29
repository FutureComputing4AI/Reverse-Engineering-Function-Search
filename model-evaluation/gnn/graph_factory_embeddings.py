##############################################################################
#                                                                            #
#  Code for the USENIX Security '22 paper:                                   #
#  How Machine Learning Is Solving the Binary Function Similarity Problem.   #
#                                                                            #
#  MIT License                                                               #
#                                                                            #
#  Copyright (c) 2019-2022 Cisco Talos                                       #
#                                                                            #
#  Permission is hereby granted, free of charge, to any person obtaining     #
#  a copy of this software and associated documentation files (the           #
#  "Software"), to deal in the Software without restriction, including       #
#  without limitation the rights to use, copy, modify, merge, publish,       #
#  distribute, sublicense, and/or sell copies of the Software, and to        #
#  permit persons to whom the Software is furnished to do so, subject to     #
#  the following conditions:                                                 #
#                                                                            #
#  The above copyright notice and this permission notice shall be            #
#  included in all copies or substantial portions of the Software.           #
#                                                                            #
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,           #
#  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF        #
#  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND                     #
#  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE    #
#  LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION    #
#  OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION     #
#  WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.           #
#                                                                            #
#  Gated Graph Sequence Neural Networks (GGSNN) and                          #
#    Graph Matching Networks (GMN) models implementation.                    #
#                                                                            #
#  This implementation contains code from:                                   #
#  https://github.com/deepmind/deepmind-research/blob/master/                #
#    graph_matching_networks/graph_matching_networks.ipynb                   #
#    licensed under Apache License 2.0                                       #
#                                                                            #
##############################################################################

import json
import math
import numpy as np
import pandas as pd
import networkx as nx

from .graph_factory_base import GraphFactoryBase
from .graph_factory_utils import *
from collections import defaultdict
from random import Random
from tqdm import tqdm

import logging
log = logging.getLogger('gnn')

class GraphFactoryEmbeddings():

    def __init__(self, func_path, feat_path, batch_size,
                 use_features, features_type, bb_features_size):
        """
            Args:
                func_path: CSV file with function pairs
                feat_path: JSON file with function features
                batch_size: size of the batch for each iteration
                use_features: if True, load the graph node features
                features_type: used to select the appropriate decoder and data
                bb_features_size: number of features at the basic-block level
        """
        
        if batch_size <= 0 or batch_size % 2 != 0:
            raise SystemError("Batch size must be even and >= 0")

        self._batch_size = batch_size
        log.info("Batch size (training): {}".format(self._batch_size))

        self._use_features = use_features
        self._features_type = features_type
        self._bb_features_size = bb_features_size
        self._decoder = str_to_scipy_sparse

        self._load_data(func_path, feat_path)

        # For reproducibility
        # Do not change the seed
        self._random = Random()
        self._random.seed(11)
        # self._np_random_state = np.random.RandomState(11)

        # Initialize the iterator
        self.next_fn_idx = 0

        # Number of functions per epoch (all of them)
        self._num_funcs = len(self._df_func)
        log.info("Tot number of functions: {}".format(
            self._num_funcs * 2))
        
        self._num_batches_in_epoch = math.floor(self._num_funcs / (self._batch_size))
        log.info("Num batches in epoch (training): {}".format(self._num_batches_in_epoch))

        return

    def _load_data(self, func_path, feat_path):
        """
        Load the training data (functions and features)

        Args
            func_path: CSV file with training functions
            feat_path: JSON file with function features
        """
        # Load CSV with the list of functions
        log.debug("Reading {}".format(func_path))
        # Read the CSV and reset the index
        self._df_func = pd.read_csv(func_path, index_col=0)
        self._df_func.reset_index(drop=True, inplace=True)

        # Load the JSON with functions features
        log.debug("Loading {}".format(feat_path))
        with open(feat_path) as gfd_in:
            self._fdict = json.load(gfd_in)

        # remove entries where feature extraction failed
        self._df_func = self._df_func[self._df_func['idb_path'].isin(self._fdict.keys())]

        # assign function names (labels) to unique integers
        names = self._df_func['func_name']
        names_pd_categorical = pd.Categorical(names)
        self.labels = names_pd_categorical.codes.tolist()

    def get_batches(self):
        """Yields batches of function data."""
        for _ in tqdm(range(self._num_batches_in_epoch), total=self._num_batches_in_epoch):
            
            batch_graphs = list()
            batch_features = list()
            batch_labels = list()

            end_fn_idx = min(self.next_fn_idx + self._batch_size, len(self._df_func))

            for fn_idx in range(self.next_fn_idx, end_fn_idx):
                
                graph, features = self._get_graph_and_features(fn_idx)
                label = self.labels[fn_idx]
                
                batch_graphs.append(graph)
                batch_features.append(features)
                batch_labels.append(label)

            self.next_fn_idx = end_fn_idx

            # Pack everything in a graph data structure
            packed_graphs = pack_batch(batch_graphs,
                                       batch_features,
                                       self._use_features,
                                       nofeatures_size=self._bb_features_size)
            labels = np.array(batch_labels, dtype=np.int32)

            yield packed_graphs, labels

    def _get_graph_and_features(self, fn_idx):
        
        idb, fva = self._df_func.iloc[fn_idx][['idb_path', 'fva']]
        graph = nx.DiGraph(str_to_scipy_sparse(self._fdict[idb][fva]['graph']))
        
        if self._use_features:
            features = self._decoder(self._fdict[idb][fva][self._features_type])
        else:
            features = -1

        return graph, features

