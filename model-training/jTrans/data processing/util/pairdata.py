import os
from collections import defaultdict
from tqdm import tqdm
from shutil import copy
import pickle
from functools import reduce
import networkx as nx
import glob
import shutil

def pairdata(data_dir):
    def get_prefix(path): # get proj name
        return os.path.basename(path).split("-")[-1].split("_")[0]

    proj2file = defaultdict(list) # proj to filename list
    for f in glob.glob(f"{data_dir}/**/*", recursive=1):
        if os.path.isfile(f):
            proj2file[get_prefix(f)].append(f)

    print("data_dir", data_dir)
    for proj, filelist in tqdm(proj2file.items()):
        if not os.path.exists(os.path.join(data_dir, proj)):
            os.makedirs(os.path.join(data_dir, proj))

        binary_func_list = []
        pkl_list = []
        delfolder = 0
        for filepath in filelist:
            src = filepath
            name = os.path.basename(filepath)
            try:
                pkl = pickle.load(open(src, 'rb'))
                pkl_list.append(pkl)
                func_list = []
                for func_name in pkl:
                    func_list.append(func_name)
                print("name, len(func_list)", name, len(func_list))
                binary_func_list.append(func_list)
            except Exception as e:
                print(e)
        if not binary_func_list:
            continue
        final_index = reduce(lambda x,y : set(x) & set(y), binary_func_list)
        print('binary_func_list all', len(final_index))

        saved_index = defaultdict(list)
        for func_name in final_index:
            for pkl in pkl_list:
                if func_name in pkl:
                    saved_index[func_name].append(pkl[func_name])
        if not os.path.exists(os.path.join("trainingdata", proj)):
            os.makedirs(os.path.join("trainingdata", proj))
        saved_pickle_name = os.path.join("trainingdata", proj, 'saved_index.pkl') # pari data
        pickle.dump(dict(saved_index), open(saved_pickle_name, 'wb'))