from sys import path
path.insert(1, '../')

from jax import numpy as jnp
from utils.distance_modules import all_pairs_euclid

from utils.triplet_modules import build_triplet_mask, loss_semihard_mining
from utils.triplet_modules import MPerClassSamplerFork as MPerClassSampler

import numpy as np

import unittest
import torch
from pytorch_metric_learning.utils import common_functions as c_f

def test_build_triplet_mask():
    
    embeddings=jnp.array([5,5,6,7,8])
    labels=jnp.array([0,0,1,1,2])

    solution = np.zeros((len(labels), len(labels), len(labels)))
    solution[2][3][0] = 1
    solution[2][3][4] = 1
    solution[3][2][0] = 1
    solution[3][2][4] = 1

    assert (build_triplet_mask(embeddings, labels)==jnp.array(solution)).all()

def test_loss_semihard_mining():

    embeddings = jnp.array([6,6,5,9,10])
    labels = jnp.array([0,0,1,1,2])
    margin = 10

    assert loss_semihard_mining(embeddings, labels, margin, all_pairs_euclid).item()==9

    embeddings = jnp.array([1, 2, 10])
    labels = jnp.array([0, 0, 1])
    margin = 5

    assert loss_semihard_mining(embeddings, labels, margin, all_pairs_euclid).item()==0

class TestMPerClassSampler(unittest.TestCase):
    """
        This unit test is copied from pytorch-metric-learning:
        https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/tests/samplers/
                                                                 test_m_per_class_sampler.py
                                                                 
        Our version of MPerClassSampler does not change functionality, but just improves the
        speed of the random sampling. The pytorch-metric-learning license is included below.

        MIT License

        Copyright (c) 2019 Kevin Musgrave

        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.

    """
    def test_mperclass_sampler(self):
        batch_size = 100
        m = 5
        length_before_new_iter = 9999
        num_labels = 100
        labels = torch.randint(low=0, high=num_labels, size=(10000,))
        sampler = MPerClassSampler(
            labels=labels, m=m, length_before_new_iter=length_before_new_iter
        )
        self.assertTrue(
            len(sampler)
            == (m * num_labels) * (length_before_new_iter // (m * num_labels))
        )
        iterable = iter(sampler)
        for _ in range(10):
            x = [next(iterable) for _ in range(batch_size)]
            curr_labels = labels[x]
            unique_labels, counts = torch.unique(curr_labels, return_counts=True)
            self.assertTrue(len(unique_labels) == batch_size // m)
            self.assertTrue(torch.all(counts == m))

    def test_mperclass_sampler_with_batch_size(self):
        for batch_size in [4, 50, 99, 100, 1024]:
            for m in [1, 5, 10, 17, 50]:
                for num_labels in [2, 10, 55]:
                    for length_before_new_iter in [100, 999, 10000]:
                        fake_embeddings = torch.randn(10000, 2)
                        labels = torch.randint(low=0, high=num_labels, size=(10000,))
                        dataset = c_f.EmbeddingDataset(fake_embeddings, labels)
                        args = [labels, m, batch_size, length_before_new_iter]
                        if (
                            (length_before_new_iter < batch_size)
                            or (m * num_labels < batch_size)
                            or (batch_size % m != 0)
                        ):
                            self.assertRaises(AssertionError, MPerClassSampler, *args)
                            continue
                        else:
                            sampler = MPerClassSampler(*args)
                        iterator = iter(sampler)
                        for _ in range(1000):
                            x = []
                            for _ in range(batch_size):
                                iterator, curr_batch = c_f.try_next_on_generator(
                                    iterator, sampler
                                )
                                x.append(curr_batch)
                            curr_labels = labels[x]
                            unique_labels, counts = torch.unique(
                                curr_labels, return_counts=True
                            )
                            self.assertTrue(len(unique_labels) == batch_size // m)
                            self.assertTrue(torch.all(counts == m))

                        dataloader = torch.utils.data.DataLoader(
                            dataset,
                            batch_size=batch_size,
                            sampler=sampler,
                            drop_last=False,
                        )
                        for _ in range(2):
                            for _, curr_labels in dataloader:
                                unique_labels, counts = torch.unique(
                                    curr_labels, return_counts=True
                                )
                                self.assertTrue(len(unique_labels) == batch_size // m)
                                self.assertTrue(torch.all(counts == m))

tests = [test_build_triplet_mask, test_loss_semihard_mining]

if __name__=='__main__':
    for test in tests:
        test()

    print("build_triplet_mask and loss_semihard_mining passed! Testing MPerClassSampler...")

    unittest.main()
        
