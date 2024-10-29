import os
import optax
from functools import partial

from utils.net_modules import BinaryTransformerEncoder
from utils.datasets import pad_collate

ALL_DATASETS = ['assemblage', 'motif', 'common-libraries',
                'marcelli-dataset-1', 'binarycorp']

def get_dataset_configs(dataset):
    dc = dict()
    
    if (dataset == 'assemblage') or (dataset=='common-libraries'):
        dc['database_path'] = '/path/to/sqlite/database'
        dc['binaries_base_path'] = '/path/to/binaries'

        if dataset == 'assemblage':
            test_fn_ids_file = '/path/to/test/function/ids'
            names_to_divide = '/path/to/names/to/divide' #path or None, or 'all'

            with open(test_fn_ids_file, 'r') as f:
                test_fn_ids_str = f.read().split('\n')
                dc['ids_to_select'] = [int(fn_id) for fn_id in test_fn_ids_str if len(fn_id)>0]

            if names_to_divide not in [None, 'all']:
                with open(names_to_divide, 'r') as f2:
                    dc['names_to_divide'] = set(f2.read().split('\n'))
            else:
                dc['names_to_divide'] = names_to_divide        
    elif dataset in ['motif', 'marcelli-dataset-1', 'binarycorp']:
        dc['path_to_functions'] = '/path/where/ghidra/extracted/functions/'

        if dataset == 'motif':
            dc['metadata_file'] = '/path/to/motif_dataset.jsonl'
            dc['path_to_files'] = '/path/to/original/motif/FILES' # not the extracted function files

    return dc
        
def get_configs():
    # data configs
    dataset = 'motif' # 'assemblage', 'motif', 'common-libraries', 'marcelli-dataset-1', 'binarycorp'
    assert (dataset in ALL_DATASETS), "dataset must be one of {}".format(ALL_DATASETS)

    trim = True         # trim functions that are too long?
    trim_length = 250   # longest allowable function length (if trim = True)
    batch_size = 600    # batch size for data loader

    custom_collate = partial(pad_collate, trim=trim, trim_length=trim_length)

    # model checkpoint
    # os.path.join(ckpt_dir, ckpt_name+str(ckpt_no)) should give the full checkpoint path
    ckpt_dir = '/path/to/checkpoint/dir'
    ckpt_name = 'transformer_checkpoint_' 
    ckpt_no = 1

    # model hyperparameters, so that we can restore the model
    lr = 0.005                      # learning rate
    
    optimizer = optax.chain(optax.clip(max_delta=1.0), optax.adam(learning_rate=lr)) # optimizer

    net = BinaryTransformerEncoder()
    
    model_init_seed = 0             # randomization seed for initializing model weights
    dropout_init_seed = 1           # randomization seed for the dropout layer (still required even if it's not used)
    
    # output files
    output_dir = '/path/to/output/dir'
    embeddings_file = os.path.join(output_dir,
                    '_'.join(['embeddings', dataset, ckpt_name + str(ckpt_no)])+'.data')
    labels_file = os.path.join(output_dir,
                    '_'.join(['labels', dataset, ckpt_name + str(ckpt_no)])+'.data')

    dataset_configs = get_dataset_configs(dataset)

    return locals()
    
