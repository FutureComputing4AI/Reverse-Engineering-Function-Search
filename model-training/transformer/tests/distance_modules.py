from sys import path
path.insert(1, '../')

from jax import numpy as jnp
from utils.distance_modules import all_pairs_euclid, one_pair_euclid
from utils.distance_modules import all_pairs_cosine, one_pair_cosine

def test_one_pair_euclid():
    a = jnp.array([0, 1])
    b = jnp.array([3, 2])

    assert one_pair_euclid(a, b)==jnp.array([10])

def test_all_pairs_euclid():
    A = jnp.array([[-2, -1], [1, 2]])
    B = jnp.array([[0, 1], [1, 0]])
    C = jnp.array([2, 3])

    assert ((all_pairs_euclid(A, B))==jnp.array([[8, 10], [2, 4]])).all()
    assert ((all_pairs_euclid(C, C))==jnp.array([[0,1],[1,0]])).all()

def test_one_pair_cosine():
    a = jnp.array([1,0])
    b = jnp.array([0,1])
    c = jnp.array([-1,0])
    d = jnp.array([0,0])

    assert one_pair_cosine(a,a)==0.
    assert one_pair_cosine(a,b)==1.
    assert one_pair_cosine(a,c)==2.
    assert one_pair_cosine(a,d)==1.
    assert one_pair_cosine(d,d)==1.

def test_all_pairs_cosine():
    A = jnp.array([[1, 0],[0,0],[0,-1]])
    B = jnp.array([[1, 0],[0,1],[-1,0]])

    assert (all_pairs_cosine(A, B)==jnp.array([[0., 1., 2.],[1.,1.,1.],[1.,2.,1.]])).all()

tests = [test_one_pair_euclid, test_all_pairs_euclid,
         test_one_pair_cosine, test_all_pairs_cosine]

if __name__=='__main__':
    for test in tests:
        test()
        
    print("All tests passed!")
