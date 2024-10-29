# imports
import sys

import numpy as np
from tqdm import tqdm

import jax
from jax import numpy as jnp
from jax import jit, value_and_grad
import jax.random as jrandom

import flax
from flax.training.train_state import TrainState

import optax

# training helper functions
def run_epoch(state, trainloader, epoch_no, configs):
    """
        Train the model for one epoch.
        Prints the epoch no. and loss to configs['out_file_name'] + '.txt'
        
        If configs['train_data_parallel'] = True, parallelizes training over
        multiple gpus

        inputs:  flax train_state
                 dataloader for training set
                 epoch_no
                 configs dictionary

        returns: updated train_state
    """
    
    def train_batch(state, inputs, labels, dropout_key): 
        '''input    flax train_state, batch of inputs, and batch of labels
           returns: loss over the batch, updated train_state
        '''
        dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)
        
        def get_loss_train(params, inputs, labels): 
            embeddings = configs['net'].apply(params, inputs, rngs={"dropout": dropout_train_key})
            return configs['train_loss_fn'](embeddings, labels) 

        # compute the gradients on the given minibatch (individually on each device)
        loss, grads = value_and_grad(get_loss_train)(state.params, inputs, labels)

        # combine the loss/gradient across all devices by taking the mean
        if configs['train_data_parallel']:
            loss = jax.lax.pmean(loss, axis_name='num_gpus')
            grads = jax.lax.pmean(grads, axis_name='num_gpus')

        state = state.apply_gradients(grads=grads)
        return state, loss
    
    if configs['train_data_parallel']:
        parallel_train_batch = jax.pmap(train_batch, axis_name='num_gpus')
        state = flax.jax_utils.replicate(state)
    else:
        train_batch = jit(train_batch)

    running_loss = []

    for inputs, labels in tqdm(trainloader):

        if configs['train_data_parallel']:
            try:
                inputs = inputs.reshape(configs['num_gpus'], configs['batch_size'], -1)
                labels = labels.reshape(configs['num_gpus'], configs['batch_size'])
            except ValueError:
                print("incomplete batch, cannot reshape: skipping", file=sys.stderr)
                continue

            state, loss = parallel_train_batch(state, inputs, labels, state.key)
            loss = loss[0] # there will be a copy of the loss from each GPU
        else:
            state, loss = train_batch(state, inputs, labels, state.key)
            
        running_loss.append(loss.item())

    if configs['train_data_parallel']:
        state = flax.jax_utils.unreplicate(state)
    
    print(epoch_no, (sum(running_loss)/len(running_loss)), sep=', ')
    sys.stdout.flush()
    
    return state 
