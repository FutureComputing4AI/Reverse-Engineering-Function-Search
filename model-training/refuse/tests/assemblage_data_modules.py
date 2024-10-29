from sys import path
path.insert(1, '../')

import pefile
import numpy as np
import torch

from utils.assemblage_data_modules import *

def test_extract_function():

    pe = pefile.PE('data/test.dll', fast_load=True)
    
    fn1_locs = [(4880, 4913), (4928, 4977)]
    fn2_locs = [(5136, 5216)]

    fn1_byte_array = np.array([ 64,  83,  72, 131, 236,  32,  72, 139, 217, 232, 130, 254, 255,
       255,  72, 141,   5, 139,  30,   0,   0,  72, 137,   3,  72, 139,
       195,  72, 131, 196,  32,  91, 195,  64,  83,  72, 131, 236,  48,
       139,  68,  36, 104,  72, 139, 217, 137,  68,  36,  40, 139,  68,
        36,  96, 137,  68,  36,  32, 232,  18, 254, 255, 255,  72, 141,
         5,  75,  30,   0,   0,  72, 137,   3,  72, 139, 195,  72, 131,
       196,  48,  91, 195]).astype(np.int16)

    fn2_byte_array = np.array([72, 131, 236,  40, 133, 210, 116,  57, 131,
       234,   1, 116,  40, 131, 234,   1, 116,  22, 131, 250,   1, 116,
        10, 184,   1,   0,   0,   0,  72, 131, 196,  40, 195, 232, 154,
         5,   0,   0, 235,   5, 232, 107,   5,   0,   0,  15, 182, 192,
        72, 131, 196,  40, 195,  73, 139, 208,  72, 131, 196,  40, 233,
        15,   0,   0,   0,  77, 133, 192,  15, 149, 193,  72, 131, 196,
        40, 233,  44,   1,   0,   0]).astype(np.int16)

    assert ((extract_function(pe, fn1_locs)) == fn1_byte_array).all()
    assert ((extract_function(pe, fn2_locs)) == fn2_byte_array).all()

def test_pad_collate():

    batch = [(torch.arange(3), 0),
             (torch.arange(5), 0),
             (10*torch.arange(7), 1),
             (None, None)]

    trim_length = 5

    trim_answer = np.array([[0, 1, 2, 256, 256],
                           [0, 1, 2, 3, 4],
                           [0, 10, 20, 30, 40]])

    no_trim_answer = np.array([[0, 1, 2, 256, 256, 256, 256],
                           [0, 1, 2, 3, 4, 256, 256],
                           [0, 10, 20, 30, 40, 50, 60]])

    labels_answer = np.array([0, 0, 1])    

    batch1 = pad_collate(batch, trim=True, trim_length=trim_length)
    batch2 = pad_collate(batch, trim=False)

    assert (batch1[0]==trim_answer).all()
    assert (batch1[1]==labels_answer).all()
    
    assert (batch2[0]==no_trim_answer).all()
    assert (batch2[1]==labels_answer).all()

def test_AssemblageFunctionsDataset():

    names = ['a', 'b', 'c', 'd', 'a', 'b', 'c', 'd']
    family_ids = [1, 2, 1, 2, 1, 2, 3, 4]
    fn_id_offset = 1
    names_to_divide = set(['a', 'd'])
    ids_to_select = [1, 2, 4, 5, 6, 8]

    seed = 0

    # test 1
    labels, labels_to_names, new_ids_to_select = AssemblageFunctionsDataset._make_labels(names,
                                                family_ids, fn_id_offset, None, None)
    assert (labels==torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])).all()
    assert labels_to_names==dict([(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd')])
    assert new_ids_to_select == [1, 2, 3, 4, 5, 6, 7, 8]

    # test 2 
    labels, labels_to_names, new_ids_to_select = AssemblageFunctionsDataset._make_labels(names,
                                                family_ids, fn_id_offset, 'all', ids_to_select)
    assert (labels==torch.tensor([0, 1, 2, 0, 1, 3])).all()
    assert labels_to_names==dict([(0, '1\\a'), (1, '2\\b'), (2, '2\\d'), (3, '4\\d')])
    assert new_ids_to_select == ids_to_select

    # test 3
    labels, labels_to_names, new_ids_to_select = AssemblageFunctionsDataset._make_labels(names,
                                                family_ids, fn_id_offset, names_to_divide, ids_to_select)
    assert (labels==torch.tensor([0, 3, 1, 0, 3, 2])).all()
    assert labels_to_names==dict([(0, '1\\a'), (3, 'b'), (1, '2\\d'), (2, '4\\d')])
    assert new_ids_to_select == ids_to_select

    # test 4
    try: 
        labels, labels_to_names, new_ids_to_select = AssemblageFunctionsDataset._make_labels(names,
                                                        family_ids, fn_id_offset, list(), ids_to_select)
    except ValueError:
        pass

    # test 5
    labels = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
    ids = [1, 2, 3, 4, 5, 6, 7, 8]

    new_labels, new_ids = AssemblageFunctionsDataset._downsample_singletons(labels, ids, 0, seed)
    assert len(new_labels)==0
    assert len(new_labels)==len(new_ids)

    # test 6
    labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    ids = [1, 2, 3, 4, 5, 6, 7, 8]

    new_labels, new_ids = AssemblageFunctionsDataset._downsample_singletons(labels, ids, 0, seed)
    assert (new_labels==labels).all()
    assert new_ids==ids

    # test 7
    labels = torch.tensor([0, 1, 0, 1, 2, 3, 4, 5, 6, 7])
    ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    new_labels, new_ids = AssemblageFunctionsDataset._downsample_singletons(labels, ids, 0.5, seed)
    assert len(new_labels)==6
    assert len(new_labels)==len(new_ids)
    

tests = [test_extract_function, test_pad_collate,
         test_AssemblageFunctionsDataset]

if __name__=='__main__':
    for test in tests:
        test()

    print("All tests passed!")
