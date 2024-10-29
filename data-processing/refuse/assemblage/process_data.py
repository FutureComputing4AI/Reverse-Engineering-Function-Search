import sys
import time
import sqlite3

from sqlite_modules import *
from dataset_configs import get_configs

start = time.time()

configs = get_configs()

time_indices = get_indices(configs['database'])
print("The indices are created. This took ", time_indices, " seconds.")
sys.stdout.flush()

duplicate_fns, time_duplicates = get_duplicate_functions(configs['database'])
print("Got duplicate functions. This took ", time_duplicates, " seconds.")
sys.stdout.flush()

binary_families, time_families = get_binary_families(configs['database'])
print("Got binary families. This took ", time_families, " seconds.")
sys.stdout.flush()

time_add_family_id = add_family_id(configs['database'])
print("Added family ids to the binaries table in the database. \
       This took ", time_add_family_id, " seconds.")
sys.stdout.flush()

time_add_fn_sizes = add_function_sizes(configs['database'])
print("Added function sizes to the functions table in the database. \
       This took ", time_add_fn_sizes, " seconds.")
sys.stdout.flush()

names_to_divide, time_lbf = determine_names_to_divide(configs['database'],
               configs['size_threshold'], configs['stdev_threshold'], configs['name_length_threshold'])

dfile = open(configs['fn_names_to_divide_file'], 'w')
print('\n'.join(names_to_divide), file=dfile)
dfile.close()

print("The names_to_divide list was created and is stored in the file ", configs['fn_names_to_divide_file'])
print("This took ", time_lbf, " seconds. ")
sys.stdout.flush()

top_funcs, time_top_funcs = get_top_funcs(configs['database'])
print("Got top functions. This took ", time_top_funcs, " seconds.")
sys.stdout.flush()

train_set_fns, test_set_fns = train_test_split(configs['database'], binary_families, top_funcs,
                                               duplicate_fns, configs['train_set_length'], configs['seed'])


train_file = open(configs['train_fns_file'], 'w')
print('\n'.join(list(map(str, train_set_fns))), file=train_file)
train_file.close()

test_file = open(configs['test_fns_file'], 'w')
print('\n'.join(list(map(str, test_set_fns))), file=test_file)
test_file.close()

end = time.time()

print("The dataset has been created, this took ", end-start, " seconds.")

if configs['config_file'] is not None:
    with open(configs['config_file'], 'w') as out_file:
        print(configs, file=out_file)
    
