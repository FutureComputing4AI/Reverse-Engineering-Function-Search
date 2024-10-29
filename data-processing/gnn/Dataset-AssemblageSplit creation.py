#!/usr/bin/env python
# coding: utf-8

##############################################################################
#                                                                            #
#  Code for the USENIX Security '22 paper:                                   #
#  How Machine Learning Is Solving the Binary Function Similarity Problem.   #
#                                                                            #
#  MIT License                                                               #
#                                                                            #
#  Copyright (c) 2019-2022 Cisco Talos                                       #
#                                                                            #
#  Permission is hereby granted, free of charge, to any person obtaining     #
#  a copy of this software and associated documentation files (the           #
#  "Software"), to deal in the Software without restriction, including       #
#  without limitation the rights to use, copy, modify, merge, publish,       #
#  distribute, sublicense, and/or sell copies of the Software, and to        #
#  permit persons to whom the Software is furnished to do so, subject to     #
#  the following conditions:                                                 #
#                                                                            #
#  The above copyright notice and this permission notice shall be            #
#  included in all copies or substantial portions of the Software.           #
#                                                                            #
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,           #
#  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF        #
#  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND                     #
#  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE    #
#  LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION    #
#  OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION     #
#  WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.           #
#                                                                            #
#  Dataset-1 creation                                                        #
#                                                                            #
##############################################################################

import json
import os
import pandas as pd
import sqlite3
from collections import defaultdict
from tqdm import tqdm

# where to save the new dataset
OUTPUT_DIR = "."

# other configs to update
split = 'Train' # 'Test' or 'Train'
CSV_FLOWCHART_FP = "flowchart_Dataset-Assemblage{}.csv".format(split)
split_fns_file = open('/path/to/assemblage/split/fn/ids.txt', 'r')
database = '/path/to/assemblage/sqlite/database'
dataset_path = '/path/to/Assemblage/split/IDBs/' # prefix must match the 'idb_path' column of CSV_FLOWCHART_FP

# read in test function IDs from Assemblage
split_fn_ids = split_fns_file.read().splitlines()
split_fn_ids = [int(x) for x in split_fn_ids]
split_fns_file.close()

# teach the CSV parser how to handle file, function names with commas
def names_w_strings(entries):
     
    # figure out how many entries are part of the file/function name, respectively
    set_file_name_entries = False
    set_func_name_entries = False
    
    for i, entry in enumerate(entries):
        if entry.startswith('0x'):
            if not set_file_name_entries:
                file_name_entries = (0, i)
                set_file_name_entries = True
            elif not set_func_name_entries:
                func_name_entries = (file_name_entries[1]+1, i)
                set_func_name_entries = True
    
    full_file_name = ",".join(entries[file_name_entries[0]:file_name_entries[1]])
    full_fn_name = ",".join(entries[func_name_entries[0]:func_name_entries[1]]) 
    
    new_entries = [full_file_name, entries[file_name_entries[1]], full_fn_name, entries[-5], 
                   entries[-4], entries[-3], entries[-2], entries[-1]]
    return new_entries

# Read the list of functions from the output of IDA flowchart
df = pd.read_csv(CSV_FLOWCHART_FP, engine='python', on_bad_lines=names_w_strings)
print(f"Shape: {df.shape}")

del df['bb_list']

# Get function IDs from Assemblage database
connection = sqlite3.connect(database)
cursor = connection.cursor()

# map idb_paths to Assemblage binary IDs
id_lookup = {}
ids_paths = cursor.execute("SELECT id, path FROM binaries").fetchall()

for bin_id, path in ids_paths:
    idb_path = os.path.join(dataset_path, path.replace('/', '_') + '.i64')
    id_lookup[idb_path] = bin_id

# get function IDs
func_info_query = "SELECT f.id FROM \
                   functions f JOIN rvas r ON f.id=r.function_id \
                   WHERE f.binary_id = {} AND \
                   r.start >= {} AND r.end <= {}"

assemblage_ids_list = []
for idx, row in tqdm(df.iterrows()):
    idb_path = row['idb_path']
    start = row['start_rva']
    end = row['end_rva']
    bin_id = id_lookup[idb_path]
    result = cursor.execute(func_info_query.format(bin_id, start, end)).fetchone()
    if result is None:
        assemblage_id = None
    else:
        assemblage_id = result[0]
        
    assemblage_ids_list.append(assemblage_id)

# Add compilation variables to the DataFrame
df['assemblage_id'] = assemblage_ids_list
df.assemblage_id = pd.Series(assemblage_ids_list, dtype = pd.Int64Dtype())

# prune out functions where function ID is not in the Assemblage test set
df_split = df[df['assemblage_id'].isin(split_fn_ids)]
print(f"Shape with only Assemblage split set functions: \t\t{df_split.shape}")

# Reset indexes
df_split.reset_index(inplace=True, drop=True)

# Save the "selected functions" to a CSV.
df_split.to_csv(os.path.join(OUTPUT_DIR, "Dataset-Assemblage{}.csv".format(split)))

# Save the "selected functions" to a JSON.
# This is useful to limit the IDA analysis to some functions only.

fset = set([tuple(x) for x in df_split[['idb_path', 'fva']].values])
print("{}: {} functions".format("all", len(fset)))

selected_functions = defaultdict(list)
for t in fset:
    selected_functions[t[0]].append(int(t[1], 16))
        
# Test
assert(sum([len(v) for v in selected_functions.values()]) == len(fset))

# Save to file
with open(os.path.join(OUTPUT_DIR, "selected_Dataset-Assemblage{}.json".format(split)), "w") as f_out:
    json.dump(selected_functions, f_out)
