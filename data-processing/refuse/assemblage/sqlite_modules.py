import sys
import os
import time
import sqlite3
import itertools
from itertools import chain
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from statistics import mean, stdev

def get_indices(database_path):
    """
        create indices on the database to improve query speed

        inputs:   path to database
        returns:  the function running time (float)
    """
    start = time.time()
    
    # connect to database
    connection = sqlite3.connect(database_path)
    connection.row_factory = lambda cursor, row: row[0]
    cursor = connection.cursor()

    # see what indices already exist, create indices where necessary
    indices = cursor.execute("""SELECT name FROM sqlite_master WHERE type='index';""").fetchall()
    if not ('rvas_by_fn' in indices):
        cursor.execute("CREATE INDEX rvas_by_fn ON rvas(function_id)")
    if not ('fns_by_binary_id' in indices):
        cursor.execute("CREATE INDEX fns_by_binary_id ON functions(binary_id)")

    connection.commit()
    cursor.close()
    connection.close()
    end = time.time()

    return end-start


def get_duplicate_functions(database_path):
    """
        inputs:  path to database
        returns: a list of function IDs that are duplicated hashes, so we can deduplicate the dataset
                 the function running time (float)
    """
    start = time.time()

    # connect to database
    connection = sqlite3.connect(database_path)
    # leave one copy of the function, mark others as duplicates
    connection.row_factory = lambda cursor, row: tuple(map(int, row[0].split(',')))[1:]
    
    cursor = connection.cursor()

    duplicates = cursor.execute("SELECT duplicates FROM \
                                 (SELECT GROUP_CONCAT(id) as duplicates, COUNT(id) as cnt \
                                 FROM functions GROUP BY hash) \
                                 WHERE cnt>1").fetchall()

    duplicate_fn_ids = set(itertools.chain.from_iterable(duplicates))

    cursor.close()
    connection.close()
    end = time.time()

    return duplicate_fn_ids, end-start        


def get_binary_families(database_path):
    """
        family = list of binaries built from the same source code, but with different compiler options

        inputs:     path to database
        returns:    a list of families with at least two binaries
                    the function running time (float)

    """
    start = time.time()
    # connect to database
    connection = sqlite3.connect(database_path)
    connection.row_factory = lambda cursor, row: tuple(map(int, row[0].split(',')))
    cursor = connection.cursor()

    binary_families = cursor.execute("SELECT family FROM \
                                     (SELECT GROUP_CONCAT(id) as family, COUNT(id) as size \
                                     FROM binaries GROUP BY github_url, file_name) \
                                     WHERE size>1").fetchall()
    
    cursor.close()
    connection.close()
    end = time.time()
    
    return binary_families, end-start

def add_family_id(database_path):
    """
        convenience function to add binary "family" ID as a column to the binaries table.
        family = group of binaries built from the same source code, but with different compiler options

        inputs:  path to database
        returns: function running time (float)
    """
    start = time.time()
    # open database connection
    connection = sqlite3.connect(database_path)

    # check if family ID has already been added
    connection.row_factory = lambda cursor, row: row[1]
    cursor = connection.cursor()

    columns = cursor.execute("PRAGMA table_info(binaries)").fetchall()
    if 'family_id' in columns:
        print("family_id column found, not regenerating.")

        cursor.close()
        connection.close()
        
        end = time.time()
        return end-start
    
    # if 'family_id' has not been created, continue on
    connection.row_factory = lambda cursor, row: tuple(map(int, row[0].split(',')))
    cursor = connection.cursor()

    # get binary families
    query = "SELECT family FROM \
             (SELECT GROUP_CONCAT(id) as family, COUNT(id) as size \
             FROM binaries GROUP BY github_url, file_name)"

    binary_families = cursor.execute(query)

    # generate reverse mapping
    fam_ids_and_bin_ids = []

    for family_id, family in enumerate(binary_families):
        for binary_id in family:
            fam_ids_and_bin_ids.append((family_id, binary_id))

    # update table
    cursor.execute("ALTER TABLE binaries ADD COLUMN family_id")
    cursor.executemany("UPDATE binaries SET family_id=? WHERE id=?", fam_ids_and_bin_ids)
    connection.commit()

    # close database connection
    cursor.close()
    connection.close()
    end = time.time()

    return end-start

def get_top_funcs(database_path, p=0.5):
    """
        inputs:  path to database,  p = threshold percentage
        returns: a list of function names that appear in more than p% of all binaries
                 the function running time (float)
    """
    start = time.time()

    assert type(p) is float, "p must be a float, has type {}".format(type(p))
    assert ((p >= 0.0) and (p <= 1.0)), "p must be a valid percentage in [0,1], p is {}".format(p) 
    
    # connect to database
    connection = sqlite3.connect(database_path)
    connection.row_factory = lambda cursor, row: row[0]    
    cursor = connection.cursor()

    num_binaries = cursor.execute("SELECT COUNT(id) FROM binaries").fetchone()

    query = "SELECT nm FROM ( \
             SELECT name as nm, COUNT(name) as cnt FROM functions \
             GROUP BY name) \
             WHERE cnt > {}".format((num_binaries * p))
    
    top_funcs_names = cursor.execute(query).fetchall()
    
    top_funcs_names = list(top_funcs_names)

    cursor.close()
    connection.close()
    end = time.time()
    
    return top_funcs_names, end-start


def add_function_sizes(database_path):
    """
        use function start and end information in Assemblage to compute function sizes
        record function sizes in a new column, "size", in the functions table

        inputs:  path to database
        returns: the function running time

    """
    start = time.time()
    # open database connection
    connection = sqlite3.connect(database_path)

    # check if function size has already been added
    connection.row_factory = lambda cursor, row: row[1]
    cursor = connection.cursor()

    columns = cursor.execute("PRAGMA table_info(functions)").fetchall()
    if 'size' in columns:
        print("size column found, not regenerating.")

        cursor.close()
        connection.close()
        
        end = time.time()
        return end-start

    # if size column does not already exist, continue on
    connection.row_factory = lambda cursor, row: row
    cursor = connection.cursor()

    # get function locations
    query = "SELECT function_id, GROUP_CONCAT(start), GROUP_CONCAT(end) \
             FROM rvas GROUP BY function_id ORDER BY function_id"

    func_ids_and_locations = cursor.execute(query)

    # compute function sizes
    sizes_and_func_ids = []

    for func_id, starts, ends in func_ids_and_locations:
        starts = [int(start) for start in starts.split(',')]
        ends = [int(end) for end in ends.split(',')]

        lengths = [end-start for (start, end) in zip(starts, ends)]
        size = sum(lengths)

        sizes_and_func_ids.append((size, func_id))

    # update table
    cursor.execute("ALTER TABLE functions ADD COLUMN size")
    cursor.executemany("UPDATE functions SET size=? WHERE id=?", sizes_and_func_ids)
    connection.commit()

    # close database connection
    cursor.close()
    connection.close()
    end = time.time()

    return end-start


def determine_names_to_divide(database_path, size_threshold, stdev_threshold,
                              name_length_threshold):
    """
        Across different source projects, programmers may repeat function names for semantically distinct
        functions. E.g. almost all projects have semantically different "main" functions. For such overloaded
        function names, it is not enough to use the function name as a label, as several independent functions
        will then get the same label. Instead, labels for these functions should comprise of the function name
        plus an identifier for which source code this function was compiled from. Here, we apply
        heuristics to determine if a function name should be divided into multiple different labels. For further
        explanation as to the source of these heuristics, please reference our paper.

        -------------------------------

        names_to_divide = set()

        for function_name n:
            1. len(n) < name_length_threshold
            2. the size of the largest function with name n > size_threshold
            3. the std. deviation in function sizes of functions with name n in Release mode > stdev_threshold
            4. the std. deviation in function sizes of functions with name n in Debug mode > stdev_threshold

            if 1 AND 2 AND 3 AND 4:
                names_to_divide.add(n)

        return list(names_to_divide)

        -----------------------------------
               
        inputs:  path to database
                 size threshold
                 std. deviation threshold
                 name length threshold
        returns: set of function names that should be divided into different labels based on originating source code
                 running time (float)
    """

    # parse sqlite query output
    def fill_dict(cursor, row):
        sizes_dict[row[0]][row[1]] = list(map(int, row[2].split(',')))

    # compute normalized standard deviation, handle edge cases
    def safe_stdev(sample):
        if len(sample) < 2:
            return 0.0
        else:
            return stdev(sample)/mean(sample)

    # compute maximum, handle edge cases
    def safe_max(sample):
        if len(sample) < 1:
            return 0
        else:
            return max(sample)

    start = time.time()
    sizes_dict = defaultdict(lambda: {'Debug': [], 'Release': []})
    
    # connect to database
    connection = sqlite3.connect(database_path)
    connection.row_factory = fill_dict
    cursor = connection.cursor()

    query = "SELECT f.name, b.build_mode, GROUP_CONCAT(f.size) \
             FROM binaries b JOIN functions f ON b.id==f.binary_id \
             GROUP BY f.name, b.build_mode"

    cursor.execute(query).fetchall()

    names_to_divide = set()
    for name in tqdm(sizes_dict.keys()):
        if len(name) > name_length_threshold:
            continue
        elif ((safe_max(sizes_dict[name]['Debug']) <  size_threshold) and
            (safe_max(sizes_dict[name]['Release']) < size_threshold)):
            continue
        else:
            debug_stdev = safe_stdev(sizes_dict[name]['Debug'])
            release_stdev = safe_stdev(sizes_dict[name]['Release'])

            if ((debug_stdev < stdev_threshold) and (release_stdev < stdev_threshold)):
                continue
            else:
                names_to_divide.add(name)

    end = time.time()

    return list(names_to_divide), end-start

def train_test_split(database_path, families_list, ubiquitous_fns, duplicate_fns, train_set_length=0.8, seed=12):
    '''
        splits functions into a train and test set, assigning entire families to either the train or test set.
        deduplicates by function hash, and also ensure common functions don't occur across splits

        this function assumes binary families are roughly the same size;

        inputs:  path to database
                 list of binary families with at least two members (from get_binary_families)
                 list of function names that occur in many binaries, regardless of source (from get_top_funcs)
                 set of duplicate function ids (from get_duplicate_functions)
                 percent of binary families to be assigned to the train set
                 random seed for creating splits

        returns: list of train set fn ids
                 list of test set fn ids

        '''

    connection = sqlite3.connect(database_path)
    connection.row_factory = lambda cursor, row: row[0]
    cursor = connection.cursor()

    if type(train_set_length) is float:
        assert((0 <= train_set_length) and (train_set_length <=1)), "percent must be between 0 and 1"
        train_set_length_int = int(train_set_length*len(families_list))
    else:
        raise TypeError("train_set_length must be a percentage (e.g. 0.8)")
    
    rng = np.random.default_rng(seed=seed)
    rng.shuffle(families_list) 
    rng.shuffle(ubiquitous_fns) 
    
    train_set_binaries_nested = families_list[0:train_set_length_int]
    test_set_binaries_nested = families_list[train_set_length_int:]

    train_set_binaries = list(chain.from_iterable(train_set_binaries_nested))
    test_set_binaries = list(chain.from_iterable(test_set_binaries_nested))

    banned_fns_train_names = ubiquitous_fns[0:int(len(ubiquitous_fns)/2)]
    banned_fns_test_names = ubiquitous_fns[int(len(ubiquitous_fns)/2):]

    train_set_fn_ids = cursor.execute(
             "SELECT id FROM functions WHERE binary_id IN {} AND name NOT IN {}"
             .format(str(tuple(train_set_binaries)), str(tuple(banned_fns_train_names)))
             ).fetchall()

    train_set_fn_ids = list(set(train_set_fn_ids).difference(duplicate_fns))

    test_set_fn_ids = cursor.execute(
             "SELECT id FROM functions WHERE binary_id IN {} AND name NOT IN {}"
             .format(str(tuple(test_set_binaries)), str(tuple(banned_fns_test_names)))
             ).fetchall()

    test_set_fn_ids = list(set(test_set_fn_ids).difference(duplicate_fns))

    cursor.close()
    connection.close()
        
    return train_set_fn_ids, test_set_fn_ids
