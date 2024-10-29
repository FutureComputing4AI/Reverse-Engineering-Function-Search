from functools import partial, update_wrapper

from utils.assemblage_data_modules import pad_collate
from utils.net_modules import BinaryTransformerEncoder
from utils.triplet_modules import loss_semihard_mining
from utils.distance_modules import all_pairs_euclid, all_pairs_cosine

import jax
import optax

def set_configs():
    # experiment variables
    out_file_dir = '/absolute/path/to/results/directory/' # must be absolute path
    out_file_name = 'transformer_training_MMDDYYYY.txt'
    experiment_description = 'optional description, gets recorded in output file'

    # paths to data variables
    database = '/path/to/assemblage/sqlite/database'
    dataset_path = '/path/to/assemblage/binaries'
    
    train_fn_ids_file = '../../data/train_fn_ids.txt'
    # fn_names_to_divide_file can be a path to a file, 'all', or None
    fn_names_to_divide = '../../data/fn_names_to_divide.txt' 

    # dataset configs
    percent_singletons=0.05         # maximum percent of the dataset that can be singleton functions
                                    # set to None to keep all singletons
    singleton_seed=50               # randomization seed for downsampling singletons

    # checkpoint settings
    ckpt_keep_every = 4             # how often to save checkpoint
    ckpt_dir = '/absolute/path/to/checkpoint/dir'     # directory to save checkpoints, must be an absolute path
    
    # training regimen hyperparameters
    trim = True                     # trim functions that are too long (boolean)
    trim_length = 250               # max function length (if trim=True)
    custom_collate = partial(pad_collate, trim=trim, trim_length=trim_length)

    epochs = 40                     # total number of epochs
    batch_size = 60                 # number of functions per batch
    batches_per_epoch = 16666        # number of batches per epoch
    len_b4_new_dataloader = batch_size * batches_per_epoch
    dataloader_seed = 70            # randomization seed for dataloader
    
    train_data_parallel = True      # parallelize training over multiple GPUs (boolean)
    num_gpus = jax.local_device_count()
    print(f"Found {num_gpus} GPUs")

    dist_fn = all_pairs_cosine      # distance function for evaluating embeddings
    margin = 0.2                    # triplet margin
    train_loss_fn = partial(loss_semihard_mining, margin=margin, dist_fn=dist_fn)

    lr = 0.005                      # learning rate

    # optimizer 
    optimizer = optax.chain(optax.clip(max_delta=1.0), optax.adam(learning_rate=lr))

    # make sure train_loss_fn and custom_collate inherit .__name__ attribute
    update_wrapper(train_loss_fn, loss_semihard_mining)
    update_wrapper(custom_collate, pad_collate)

    # Model hyperparameters are configured in the net_modules `TransformerConfig`
    net = BinaryTransformerEncoder()
    
    model_init_seed = 0             # randomization seed for initializing model weights
    dropout_init_seed = 1           # randomization seed for encoder dropout layer

    return locals()
