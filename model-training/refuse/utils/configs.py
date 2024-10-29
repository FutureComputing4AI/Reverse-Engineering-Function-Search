from functools import partial, update_wrapper

from utils.assemblage_data_modules import pad_collate
from utils.net_modules import REFUSE
from utils.triplet_modules import loss_semihard_mining
from utils.distance_modules import all_pairs_euclid, all_pairs_cosine

import jax
import optax

def set_configs():
    # experiment variables
    out_file_dir = '/absolute/path/to/results/directory/' # must be absolute path
    out_file_name = 'refuse_training_MMDDYYYY.txt'
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

    epochs = 1                      # total number of epochs
    batch_size = 600                # number of functions per batch
    batches_per_epoch = 10           # number of batches per epoch
                                    # 16666 was chosen to make len_b4_new_dataloader=10M
    len_b4_new_dataloader = batch_size * batches_per_epoch
    dataloader_seed = 70            # randomization seed for dataloader
    
    train_data_parallel = False      # parallelize training over multiple GPUs (boolean)
    num_gpus = jax.local_device_count()

    dist_fn = all_pairs_cosine      # distance function for evaluating embeddings
    margin = 0.2                    # triplet margin
    train_loss_fn = partial(loss_semihard_mining, margin=margin, dist_fn=dist_fn)

    lr = 0.005                      # learning rate

    # optimizer 
    optimizer = optax.chain(optax.clip(max_delta=1.0), optax.adam(learning_rate=lr))

    # make sure train_loss_fn and custom_collate inherit .__name__ attribute
    update_wrapper(train_loss_fn, loss_semihard_mining)
    update_wrapper(custom_collate, pad_collate)

    # model hyperparameters
    final_embd_size = 128           # size of final embedding
    window_size = 8,                # convolutional layer window size
    stride = 8                      # convolutional layer stride
    byte_embd_size = 8              # size of learned embedding for individual bytes 0, 1, ..., 255
    log_stride = None               # for large stride, can pass log_2(stride) here instead
    
    net = REFUSE(channels=final_embd_size, window_size=window_size, 
                 stride=stride, embd_size=byte_embd_size, log_stride=log_stride)
    
    model_init_seed = 0             # randomization seed for initializing model weights

    return locals()
