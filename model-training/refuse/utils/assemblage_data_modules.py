import sys
import os

import pefile
import sqlite3

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

def extract_function(pe, func_locations):
    '''
        extract function bytes from pe file, given RVA locations of function

        inputs:  pe is a pefile.PE object
                 func_locations is a list of (rva_start, rva_end) tuples
                 rva_start/end should be ints
        returns: numpy byte array
    '''
     
    func_data_list = []

    for (rva_start, rva_end) in func_locations:
        try:
            data = pe.get_data(rva=rva_start, length=(rva_end-rva_start))
        except pefile.PEFormatError as error:
            return error 
        func_data_list.append(data)
        
    func_data = b''.join(x for x in func_data_list)
    func_array = np.frombuffer(func_data, dtype=np.uint8).astype(np.int16)   
    return func_array 

def pad_collate(batch, trim=True, trim_length=250):
    ''' deal with possibly unequal length of functions by padding out shorter functions
        pad everything to length trim

        input:   list of (function, label) tuples where function is a pytorch tensor, label is an integer
        returns: tuple of (all functions, all labels), where both elements of the tuple are numpy arrays
    '''

    batch_fns, batch_labels = zip(*batch)

    # remove Nones (corrupted functions) from the batch
    batch_fns = list(filter(lambda x: x is not None, batch_fns))
    batch_labels = list(filter(lambda x: x is not None, batch_labels))
    
    if trim:
        batch_fns.append(torch.zeros((trim_length))) # make sure batch is padded to at least trim_length
    
    batch_labels = torch.tensor(batch_labels)
    
    padded_data = torch.nn.utils.rnn.pad_sequence(batch_fns, batch_first=True, padding_value=256.0) 
    
    # if 'trim' and function is more than 'trim_length' bytes, only keep the first 'trim_length'
    if trim:
        if padded_data.shape[-1] > trim_length:
            padded_data = padded_data[:,:trim_length]
        padded_data = padded_data[:-1] # remove extra tensor added to fix padding size

    assert len(padded_data)==len(batch_labels)

    return padded_data.detach().numpy(), batch_labels.detach().numpy()

class AssemblageFunctionsDataset(Dataset):
    """
        Dataset of Assemblage Functions

        constructor arguments:

            database_path:       path to Assemblage sqlite database
            binaries_base_path:  path to Assemblage binaries
            ids_to_select:       a list or tuple consisting of a subset of all the function_ids in the
                                 'functions' table. Default (None) will use every function_id.
            percent_singletons:  what is the maximum percent of the labels that should be singletons (labels that only
                                 occur once across all functions in the dataset.) Should be a float between [0, 1).
                                 Default value 'None' results in no downsampling of singletons.
            singleton_seed:      random seed for downsampling singletons
            names_to_divide:     set of function names that should be split into multiple labels each, based on the
                                 originating source code. Can be type None, string 'all', or a set of function names

        get_name: takes an integer label and returns the function name (string) associated with that label
        
    """

    def __init__(self, database_path='data.sqlite', binaries_base_path='dataset', ids_to_select=None,
                 percent_singletons=None, singleton_seed=50, names_to_divide = None):

        # connect to database
        self.connection = sqlite3.connect(database_path)
        self.connection.row_factory = lambda cursor, row: row[0]
        self.cursor = self.connection.cursor()
        
        self.binaries_base_path = binaries_base_path
        self.ids_to_select = ids_to_select

        # create indexes if they don't exist
        indices = self.cursor.execute("""SELECT name FROM sqlite_master WHERE type='index';""").fetchall()
        
        if not ('rvas_by_fn' in indices):
            self.cursor.execute("CREATE INDEX rvas_by_fn ON rvas(function_id)")

        self.connection.row_factory = lambda cursor, row: row
        self.cursor = self.connection.cursor()

        # get the names of the functions in this dataset, to make labels
        select_fns_query = "SELECT f.name, b.family_id \
                            FROM binaries b JOIN functions f ON b.id==f.binary_id \
                            ORDER BY f.id"
        names, family_ids  = zip(*self.cursor.execute(select_fns_query).fetchall())

        fn_id_offset = self.cursor.execute("SELECT MIN(id) FROM functions").fetchone()[0]

        self.labels, self.labels_to_names, self.ids_to_select = AssemblageFunctionsDataset._make_labels(names,
                                                                    family_ids, fn_id_offset, names_to_divide, self.ids_to_select)

        if percent_singletons is not None:
            self.labels, self.ids_to_select = AssemblageFunctionsDataset._downsample_singletons(self.labels,
                                                self.ids_to_select, percent_singletons, singleton_seed)

        # pytorch sampler will return indexes corresponding to the position of the label it selected in self.labels
        # this index may not correspond with the function_id of that function, so we track this mapping explicitly
        self.dataset_id_to_function_id = dict(zip([i for i in range(len(self.labels))], self.ids_to_select))

    @staticmethod
    def _make_labels(names, family_ids, fn_id_offset, names_to_divide, ids_to_select):

        def get_label(name, family_id):
            if name in names_to_divide:
                return str(family_id) + '\\' + name
            else:
                return name

        if ids_to_select is None:
            ids_to_select = [i+fn_id_offset for i in range(len(names))]

        if names_to_divide is None:
            names = [names[i-fn_id_offset] for i in ids_to_select]
        elif names_to_divide == 'all':
            names = [str(family_ids[i-fn_id_offset]) + '\\' + names[i-fn_id_offset] for i in ids_to_select]
        elif type(names_to_divide) is set:
            names = [get_label(names[i-fn_id_offset], family_ids[i-fn_id_offset]) for i in ids_to_select]
        else:
            raise ValueError("Invalid type for names_to_divide. Must be None, 'all', or a set. \
                               Got type {}.".format(type(names_to_divide)))
            
        names_pd_categorical = pd.Categorical(pd.Series(names))
        labels = torch.tensor(names_pd_categorical.codes)
        labels_to_names = dict(zip(labels.tolist(), names))
            
        return labels, labels_to_names, ids_to_select

    @staticmethod
    def _downsample_singletons(labels, ids_to_select, percent_singletons, singleton_seed):

        assert 0 <= percent_singletons < 1, "percent_singletons must be in [0, 1), or None"
        assert type(singleton_seed) is int, "singleton_seed must be an integer"
        torch.manual_seed(singleton_seed)

        # compute number of singletons to keep in downsampling
        unique_labels, counts = torch.unique(labels, return_counts=True)
        num_singletons = torch.sum(counts==1).item()
        desired_num_singletons = int((percent_singletons * (len(unique_labels)-num_singletons))/(1.0-percent_singletons))
        desired_num_singletons = min(desired_num_singletons, num_singletons)

        # randomly choose which singletons to keep
        all_singletons = unique_labels[counts==1]
        selected_singletons = torch.take(all_singletons, torch.randperm(len(all_singletons))[:desired_num_singletons])

        # downsample
        labels_to_keep = torch.cat((unique_labels[counts>1], selected_singletons))
        positions_to_keep = torch.isin(labels, labels_to_keep)

        labels = labels[positions_to_keep]
        ids_to_select = torch.tensor(ids_to_select)[positions_to_keep]
        ids_to_select = ids_to_select.detach().numpy().tolist()

        return labels, ids_to_select

    def __len__(self):

        return len(self.labels)

    def get_name(self, label):

        return self.labels_to_names[label]

    def __getitem__(self, idx):

        fn_id = self.dataset_id_to_function_id[idx]

        binary_id= self.cursor.execute("SELECT binary_id FROM functions WHERE id=?", (fn_id,)).fetchone()[0]
        path = self.cursor.execute("SELECT path FROM binaries WHERE id=?", (binary_id,)).fetchone()[0]
        func_locs = self.cursor.execute("SELECT start, end FROM rvas WHERE function_id=?", (fn_id,)).fetchall()
            
        pe = pefile.PE(os.path.join(self.binaries_base_path, path), fast_load=True)
        func_array = extract_function(pe, func_locs)
        pe.close()

        label = self.labels[idx].item()

        if type(func_array) is np.ndarray:
            return torch.from_numpy(func_array), label
        else:
            print(func_array, file=sys.stderr)
            return None, None
