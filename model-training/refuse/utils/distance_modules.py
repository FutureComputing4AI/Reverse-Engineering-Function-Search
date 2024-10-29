from jax import numpy as jnp
from jax import vmap
from jax.lax import cond

def one_pair_euclid(a, b):
    """
        input:  numpy or jax numpy vectors a, b
        output: square euclidean distance between a, b
    """
    
    return jnp.dot((a-b), (a-b))

def one_pair_cosine(a, b):
    """
        input:  numpy or jax numpy vectors a, b
        output: cosine distance between a, b
    """
    
    num = jnp.dot(a, b)
    denom = jnp.linalg.norm(a) * jnp.linalg.norm(b)

    # in the very unlikely case one of the vectors is the zero vector, return 0
    similarity = cond(denom==0, lambda x,y: 0., lambda x,y: x/y, num, denom)
        
    return 1-similarity

# vectorize distance functions above to matrices of vectors
all_pairs_euclid = vmap(vmap(one_pair_euclid, (0, None), 0),(None, 0), 1)

all_pairs_cosine = vmap(vmap(one_pair_cosine, (0, None), 0), (None, 0), 1)


