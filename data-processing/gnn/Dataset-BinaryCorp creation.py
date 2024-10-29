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

from collections import defaultdict


# where to save the new dataset
OUTPUT_DIR = "."


# The starting point
CSV_FLOWCHART_FP = "flowchart_Dataset-BinaryCorp.csv"


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

df.to_csv(os.path.join(OUTPUT_DIR, "Dataset-BinaryCorp.csv"))

# Save the "selected functions" to a JSON.
# This is useful to limit the IDA analysis to some functions only.
fset = set([tuple(x) for x in df[['idb_path', 'fva']].values])
print("{}: {} functions".format("all", len(fset)))

selected_functions = defaultdict(list)
for t in fset:
    selected_functions[t[0]].append(int(t[1], 16))
        
# Test
assert(sum([len(v) for v in selected_functions.values()]) == len(fset))

# Save to file
with open(os.path.join(OUTPUT_DIR, "selected_Dataset-BinaryCorp.json"), "w") as f_out:
    json.dump(selected_functions, f_out)
