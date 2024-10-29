import sys
import os
import math
from tqdm import tqdm
from collections import defaultdict
import pandas as pd

import numpy as np

# where the file-level BSim Results are stored
# input_folder = '/data/nkfleis/query_results'

def analysis(input_folder, motif_metadata_filepath=None):

    motif_mode = (motif_metadata_filepath is not None)

    upper_rrs = []
    lower_rrs = []

    if motif_mode:
        metadata = pd.read_json(path_or_buf=motif_metadata_filepath, lines=True)
        metadata = metadata.set_index('md5')

        # only the malware family labels
        labels = metadata['label']
        labels_dict = labels.to_dict() # key = file hash, value = malware family label

    print("Gathering all of the query outputs. For large datasets, this may take a while...")
    all_files = [os.path.join(root, file) for root, _, files in os.walk(input_folder) for file in files]
    failures = []

    for file in tqdm(all_files):
        #1. read in all the bsim query results
        bsim_results = defaultdict(dict)
        with open(file, 'r') as f:
            for line in f:
                result = line.rstrip().split('\\')
                query, match, sim = result[0], result[1], float(result[2])

                if motif_mode:
                    try:
                        query = labels_dict[query]
                        match = labels_dict[match]
                    except KeyError:
                        failures.append((result, file))
                        continue
            
                bsim_results[query][match] = sim

        #2. get the reciprocal rank score for each query function
        for query in bsim_results.keys():
            matches, scores = list(zip(*bsim_results[query].items()))

            # multiply by -1 so that the most similar is in position 0 instead of N
            sort_index = np.argsort(-1*np.array(scores)) 
            sorted_matches = np.array(matches)[sort_index]

            upper_rank = len(sorted_matches) + 1
            lower_rank = math.inf
            
            for r in range(len(sorted_matches)):
                if sorted_matches[r] == query:
                    upper_rank = r+1
                    lower_rank = r+1
                    break

            upper_rrs.append(1/upper_rank)
            lower_rrs.append(1/lower_rank)
                
    upper_mrr = np.mean(upper_rrs)
    lower_mrr = np.mean(lower_rrs)

    print("upper bound mrr: ", upper_mrr)
    print("lower bound mrr: ", lower_mrr)
    if len(failures) > 0: print(f"Failed to include {len(failures)} files: \n{failures}")

    return upper_mrr, lower_mrr
