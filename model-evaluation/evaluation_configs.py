import os

def get_configs():

    dataset = 'assemblage' # 'assemblage','motif','common-libraries','marcelli-dataset-1', or 'binarycorp'
    model = 'refuse' # 'refuse', 'transformer' or 'gnn'
    descriptor = ''   # optional additional descriptor for results file name

    embd_size = 128   # dimension of the embeddings
    dist_fn = 'cosine' # how to measure the distance between embeddings
                       # should be 'cosine' for REFuSe, 'euclid' for GNN
    K = 30 # number of nearest neighbors to check
    M = 32 # hnsw parameter, "no. neighbors used in the graph. A larger M is more accurate
           # but uses more memory."

    recall_ks = [1, 2, 5, 10] # where to compute recall, should be a positive int, list of positive ints, or None

    how_normalize = None      # either None (aka don't normalize), 'source', 'type', or 'all'
                              # currently, only the (Assemblage data, REFuSe model) combo supports normalization

    embeddings_file = '/path/to/saved/embeddings'
    labels_file = '/path/to/saved/labels'

    output_file = os.path.join(model, 'results', 'results_' + dataset + descriptor + '.txt')

    #############################
    # assemblage dataset variables, to do label normalization experiments
    
    dataset_configs = dict()

    if (dataset == 'assemblage') and (model in ['refuse', 'transformer']):
        dataset_configs['database_path'] = '/path/to/sqlite/database'
        dataset_configs['binaries_base_path'] = '/path/to/binaries'

        test_fn_ids_file = '/path/to/test/function/ids'
        names_to_divide = 'all'

        with open(test_fn_ids_file, 'r') as f:
            test_fn_ids_str = f.read().split('\n')
            dataset_configs['ids_to_select'] = [int(fn_id) for fn_id in test_fn_ids_str if len(fn_id)>0]

        if names_to_divide not in [None, 'all']:
            with open(names_to_divide, 'r') as f2:
                dataset_configs['names_to_divide'] = set(f2.read().split('\n'))
        else:
            dataset_configs['names_to_divide'] = names_to_divide

    return locals()


    
    
