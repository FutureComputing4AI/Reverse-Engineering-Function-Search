# imports
import sys
import time
from tqdm import tqdm

import numpy as np
from torch.utils.data import DataLoader

import jax
from jax import numpy as jnp
from jax import random as jrandom

import flax
from flax.training.train_state import TrainState
from flax.training import checkpoints

from utils.gen_embeddings_configs import get_configs
from utils.datasets import extract_function
from utils.datasets import AssemblageFunctionsDataset, MotifDataset, AssemblageFunctionsDatasetBasic, StandardGhidraFunctionsDataset 

DATASET_MAP = {'assemblage': AssemblageFunctionsDataset,
               'motif': MotifDataset,
               'common-libraries': AssemblageFunctionsDatasetBasic,
               'marcelli-dataset-1': StandardGhidraFunctionsDataset,
               'binarycorp': StandardGhidraFunctionsDataset
               }

# set up

start = time.time()

configs = get_configs()
net = configs['net']

embeddings_file = configs['embeddings_file']
labels_file = configs['labels_file']

f = open(embeddings_file, 'w')
f.close()
f = open(labels_file, 'w')
f.close()

# dataset
dataset_configs = configs['dataset_configs']

dataset = DATASET_MAP[configs['dataset']](**dataset_configs)

# instantiate model
init_rngs = {'params': jrandom.PRNGKey(configs['model_init_seed'])} 
x = jnp.expand_dims(dataset[0][0].detach().numpy(), axis=0)
init_params = net.init(init_rngs, x)
net_state = TrainState.create(apply_fn=net.apply, params=init_params, tx=configs['optimizer'])

# load the saved state from its checkpoint
net_state = checkpoints.restore_checkpoint(ckpt_dir=configs['ckpt_dir'], target=net_state,
                                            prefix = configs['ckpt_name'], step=configs['ckpt_no'])


# get embeddings at checkpoint and store in numpy.memmap file
dataloader = DataLoader(dataset, batch_size=configs['batch_size'], shuffle=False, collate_fn=configs['custom_collate'])

num_stored_embeddings = 0
embd_size = net.channels

for new_inputs, new_labels in tqdm(dataloader):

    amt_new_data = len(new_labels) 
    
    new_embeddings = net.apply(net_state.params, new_inputs)    
    embeddings = np.memmap(embeddings_file, dtype='float32', mode='r+',
                            shape=(num_stored_embeddings+amt_new_data, embd_size))
    embeddings[num_stored_embeddings:,:] = new_embeddings[:,:]
    
    labels = np.memmap(labels_file, dtype='int64', mode='r+',
                            shape=(num_stored_embeddings+amt_new_data))
    labels[num_stored_embeddings:] = new_labels[:]
        
    num_stored_embeddings += amt_new_data

end = time.time()
print("embeddings generated, this took {} seconds".format(end-start), file=sys.stderr)
