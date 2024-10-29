# imports
import sys
import os
import time

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import jax
from jax import numpy as jnp
import jax.random as jrandom

import flax
from flax.training.train_state import TrainState
from flax.training import checkpoints

import optax

from utils.assemblage_data_modules import AssemblageFunctionsDataset
from utils.triplet_modules import MPerClassSamplerFork
from utils.configs import set_configs
from utils.train_modules import run_epoch

# debug NANs
jax.config.update("jax_debug_nans", True)

########################################

# get configs and log to output file
configs = set_configs()
sys.stdout = open(os.path.join(configs['out_file_dir'], configs['out_file_name']), 'a')
print("configs")
print(configs)
print()

start = time.time()

# load data
with open(configs['train_fn_ids_file'], 'r') as f:
    train_fn_ids_str = f.read().split('\n')
    train_fn_ids = [int(fn_id) for fn_id in train_fn_ids_str if len(fn_id) > 0]

if configs['fn_names_to_divide'] in [None, 'all']:
    names_to_divide = configs['fn_names_to_divide']
else:
    with open(configs['fn_names_to_divide'], 'r') as f2:
        names_to_divide = set(f2.read().split('\n'))

print("loaded data", file=sys.stderr)
sys.stderr.flush()

# make dataset
train_set = AssemblageFunctionsDataset(configs['database'], configs['dataset_path'], train_fn_ids,
                                       configs['percent_singletons'], configs['singleton_seed'], names_to_divide)

print("made dataset", file=sys.stderr)
print("The number of functions in the training set is: ", len(train_set))
sys.stderr.flush()

# make data loader
train_sampler = MPerClassSamplerFork(train_set.labels, m=2, batch_size=configs['batch_size'],
                    length_before_new_iter=configs['len_b4_new_dataloader'], seed=configs['dataloader_seed'])

# when data parallel training, get num_gpu batches at once from dataloader
if configs['train_data_parallel']:
    dataloader_batch_size = configs['batch_size'] * configs['num_gpus']
else:
    dataloader_batch_size = configs['batch_size']

triplet_train_loader = DataLoader(train_set, batch_size=dataloader_batch_size, 
                                  sampler=train_sampler, collate_fn=configs['custom_collate'])

print("made dataloader", file=sys.stderr)
sys.stderr.flush()

# instantiate model
init_rngs = {'params': jrandom.PRNGKey(configs['model_init_seed'])} 
x = jnp.expand_dims(train_set[0][0].detach().numpy(), axis=0)
init_params = configs['net'].init(init_rngs, x)
net_state = TrainState.create(apply_fn=configs['net'].apply, params=init_params, tx=configs['optimizer'])

# train model
print("training embedding net...", file=sys.stderr)
print("----------------------------------------------------------------")
print("epoch, training loss")
sys.stderr.flush()

for epoch in tqdm(range(configs['epochs'])):
    net_state = run_epoch(net_state, triplet_train_loader, epoch, configs)

    checkpoints.save_checkpoint(ckpt_dir=configs['ckpt_dir'], target=net_state, step=epoch,
                                prefix=(configs['out_file_name'].strip('.txt')+'_'), keep=1, 
                                keep_every_n_steps=configs['ckpt_keep_every'])

end = time.time()
print("\nThe total training time (in seconds) was: ", end-start)
print("finished training!", file=sys.stderr)
