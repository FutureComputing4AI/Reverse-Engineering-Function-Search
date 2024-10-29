import sys
import os
from os import listdir
import sqlite3
import pefile
import hashlib
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

################################################################
#                       HELPER FUNCTIONS                       #
################################################################


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

def extract_function(pe, func_locations):
    '''
        For datasets built with Assemblage:

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


################################################################
#                           DATASETS                           #
################################################################

class GhidraFunctionsDatasetBase(Dataset):
    """
        Base PyTorch dataset for functions extracted by GhidraExtractFunctionBytes.java.
        Subclasses must implement _getlabel.

        path_to_functions:     path to the folder where the function files from running
                               GhidraExtractFunctionBytes.java were stored
    """

    def __init__(self, path_to_functions):
        self.fn_base_path = path_to_functions
        self.data = [f for f in listdir(self.fn_base_path)]

    def __len__(self):
        return len(self.data)
    
    def _getbytes(self, fn_file_name):
        with open(os.path.join(self.fn_base_path, fn_file_name), 'rb') as fn_file:
            fn_bytes_string = fn_file.read()
            try:
                fn_bytes = torch.frombuffer(fn_bytes_string, dtype=torch.uint8).type(torch.int16)
            except ValueError:
                fn_bytes = torch.Tensor().type(torch.int16)
            
        return fn_bytes
        
    def _getlabel(self, fn_file_name):
        raise NotImplementedError("Subclasses of GhidraFunctionsDatasetBase should implement __getlabel.")
    
    def __getitem__(self, idx):
        fn_file_name = self.data[idx]
        fn_bytes = self._getbytes(fn_file_name)
        label = self._getlabel(fn_file_name)
        
        return fn_bytes, label


class MotifDataset(GhidraFunctionsDatasetBase):
    """
        PyTorch dataset for functions from MOTIF.
        Each function is labeled according to the malware family it was found in.

        Constructor Arguments:
            path_to_functions: path to the folder where the function files from running
                               GhidraExtractFunctionBytes.java were stored
            metadata_file:     path to the motif_dataset.jsonl file, which can be downloaded from
                               here: https://github.com/boozallen/MOTIF
            path_to_files:     path to original MOTIF files (files named MOTIF_hash)

    """
    
    def __init__(self, path_to_functions, metadata_file, path_to_files):
        super().__init__(path_to_functions)
        self.labels_dict = self._create_labels_dict(metadata_file)

        self.hash_dict = {}
        for file_ in os.scandir(path_to_files):
            orig_hash = file_.name.replace('MOTIF_', '')
            new_hash = self._sha256sum(os.path.join(path_to_files, file_.name))
            self.hash_dict[new_hash] = orig_hash

    def _sha256sum(self, filename):
        with open(filename, 'rb', buffering=0) as f:
            return hashlib.file_digest(f, 'sha256').hexdigest()

    def _create_labels_dict(self, metadata_file):
        # read in all metadata
        metadata = pd.read_json(path_or_buf=metadata_file, lines=True)
        metadata = metadata.set_index('md5')

        # only the labels
        labels = metadata['label']
        labels_dict = labels.to_dict() # key = file hash, value = malware family label

        return labels_dict
        
    def _getlabel(self, fn_file_name):
        new_file_hash = fn_file_name.split('\\')[0]
        orig_file_hash = self.hash_dict[new_file_hash]
        label = self.labels_dict[orig_file_hash]
        return label

class StandardGhidraFunctionsDataset(GhidraFunctionsDatasetBase):
    """
        PyTorch dataset for functions extracted by Ghidra, where name=label

        path_to_functions: path to the folder where the function files from running
                           GhidraExtractFunctionBytes.java were stored
    """

    def __init__(self, path_to_functions):
        super().__init__(path_to_functions)

        names = [file_name.split('\\')[1] for file_name in self.data]
        
        # get a unique integer label for each function name
        names_pd_categorical = pd.Categorical(pd.Series(names))
        self.labels = torch.tensor(names_pd_categorical.codes)
        self.labels_to_names = dict(zip(self.labels.tolist(), names))
        self.names_to_labels = dict(zip(names, self.labels.tolist()))

    def _getlabel(self, fn_file_name):
        function_name = fn_file_name.split('\\')[1]
        return self.names_to_labels[function_name]

    def get_name(self, label):
        return self.labels_to_names[label]

class AssemblageFunctionsDatasetBasic(Dataset):

    """
        Simplified AssemblageFunctionsDataset for the Common Libraries Experiment.
        Every function in the sqlite database is part of the dataset
        labels = function names

        constructor arguments:

            database_path:       path to CommonLibraries sqlite database
            binaries_base_path:  path to CommonLibraries binaries
        
        get_name: takes an integer label and returns the function name (string)
                  associated with that label
        
    """

    def __init__(self, database_path='data.sqlite', binaries_base_path='dataset'):
        # connect to database
        self.connection = sqlite3.connect(database_path)
        self.connection.row_factory = lambda cursor, row: row[0]
        self.cursor = self.connection.cursor()

        self.binaries_base_path = binaries_base_path

        # create indexes if they don't exist
        indices = self.cursor.execute("""SELECT name FROM sqlite_master WHERE type='index';""").fetchall()
        
        if not ('rvas_by_fn' in indices):
            self.cursor.execute("CREATE INDEX rvas_by_fn ON rvas(function_id)")

        # make labels from function names
        select_fns_query = "SELECT name FROM functions ORDER BY id"
        names = self.cursor.execute(select_fns_query).fetchall()

        names_pd_categorical = pd.Categorical(pd.Series(names))
        self.labels = torch.tensor(names_pd_categorical.codes)
        self.labels_to_names = dict(zip(self.labels.tolist(), names))

        # reset sqlite3 cursor
        self.connection.row_factory = lambda cursor, row: row
        self.cursor = self.connection.cursor()

    def __len__(self):

        return len(self.labels)

    def get_name(self, label):

        return self.labels_to_names[label]

    def __getitem__(self, idx):

        fn_id = idx+1 # fn ids are 1-indexed in SQL

        binary_id= self.cursor.execute("SELECT binary_id FROM functions WHERE id=?", (fn_id,)).fetchone()[0]
        assemblage_path = self.cursor.execute("SELECT path FROM binaries WHERE id=?", (binary_id,)).fetchone()[0]
        func_locs = self.cursor.execute("SELECT start, end FROM rvas WHERE function_id=?", (fn_id,)).fetchall()
            
        pe = pefile.PE(os.path.join(self.binaries_base_path, assemblage_path), fast_load=True)
        func_array = extract_function(pe, func_locs)
        pe.close()

        label = self.labels[idx].item()

        if type(func_array) is np.ndarray:
            return torch.from_numpy(func_array), label
        else:
            print("Could not retrieve function {} from file {}".format(self.get_name(label), assemblage_path))
            return None, None

class AssemblageFunctionsDataset(Dataset):
    """
        Dataset of Assemblage Functions

        constructor arguments:

            database_path:       path to Assemblage sqlite database
            binaries_base_path:  path to Assemblage binaries
            ids_to_select:       a list or tuple consisting of a subset of all the function_ids in the
                                 'functions' table. Default (None) will use every function_id.
            percent_singletons:  what is the maximum percent of the dataset that should be singleton functions
                                 (functions with a label that only occurs once.) Should be a float between [0, 1).
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

        self.labels, self.labels_to_names = self._make_labels(names, family_ids, fn_id_offset, names_to_divide)

        if percent_singletons is not None:
            self._downsample_singletons(percent_singletons, singleton_seed)

        # pytorch sampler will return indexes corresponding to the position of the label it selected in self.labels
        # this index may not correspond with the function_id of that function, so we track this mapping explicitly
        self.dataset_id_to_function_id = dict(zip([i for i in range(len(self.labels))], self.ids_to_select))

    def _make_labels(self, names, family_ids, fn_id_offset, names_to_divide):

        def get_label(name, family_id):
            if name in names_to_divide:
                return str(family_id) + '\\' + name
            else:
                return name

        if self.ids_to_select is None:
            self.ids_to_select = [i+fn_id_offset for i in range(len(names))]

        if names_to_divide is None:
            names = [names[i-fn_id_offset] for i in self.ids_to_select]
        elif names_to_divide == 'all':
            names = [str(family_ids[i-fn_id_offset]) + '\\' + names[i-fn_id_offset] for i in self.ids_to_select]
        elif type(names_to_divide) is set:
            names = [get_label(names[i-fn_id_offset], family_ids[i-fn_id_offset]) for i in self.ids_to_select]
        else:
            print("Invalid type for label_by_family_set. Must be None, 'all', or a set. \
                   Got type ", type(label_by_family_set), ".", file=sys.stderr)
            sys.exit(3)
            
        names_pd_categorical = pd.Categorical(pd.Series(names))
        labels = torch.tensor(names_pd_categorical.codes)
        labels_to_names = dict(zip(labels.tolist(), names))
            
        return labels, labels_to_names

    def _downsample_singletons(self, percent_singletons, singleton_seed):

        assert 0 <= percent_singletons < 1, "percent_singletons must be in [0, 1), or None"
        assert type(singleton_seed) is int, "singleton_seed must be an integer"
        torch.manual_seed(singleton_seed)

        # compute number of singletons to keep in downsampling
        unique_labels, counts = torch.unique(self.labels, return_counts=True)
        num_singletons = torch.sum(counts==1).item()
        desired_num_singletons = int((percent_singletons * (len(unique_labels)-num_singletons))/(1.0-percent_singletons))
        desired_num_singletons = min(desired_num_singletons, num_singletons)

        # randomly choose which singletons to keep
        all_singletons = unique_labels[counts==1]
        selected_singletons = torch.take(all_singletons, torch.randperm(len(all_singletons))[:desired_num_singletons])

        # downsample
        labels_to_keep = torch.cat((unique_labels[counts>1], selected_singletons))
        positions_to_keep = torch.isin(self.labels, labels_to_keep)

        self.labels = self.labels[positions_to_keep]
        self.ids_to_select = torch.tensor(self.ids_to_select)[positions_to_keep]
        self.ids_to_select = self.ids_to_select.detach().numpy().tolist()

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
