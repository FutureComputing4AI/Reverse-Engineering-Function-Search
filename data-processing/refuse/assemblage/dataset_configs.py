def get_configs():
    """
        database:             /path/to/assemblage/sqlite/database
        train_set_length:     percent of the data in the training split
        seed:                 random seed for creating the train/test split

        # next three variables determine when to split a function name into multiple labels
        # for more info on see the docstring for "determine_names_to_divide" in
        # REFuSe-public/data-processing/assemblage/sqlite_modules.py 
          
        size_threshold:        don't divide when all functions with a given name are less than this threshold
        stdev_threshold:       don't divide when the size variance among Debug functions and among Release
                               functions with a given name are both less than this threshold
        name_length_threshold: don't divide when the length of the function name is greater than this threshold
        
        train_fns_file:          /where/to/save/train/fn/ids
        test_fns_file:           /where/to/save/test/fn/ids
        fn_names_to_divide_file: /where/to/save/fn/names/to/divide
        config_file:             /where/to/save/copy/of/configs

    """

    database = '../../../data/assemblage.sqlite'
    train_set_length = 0.8
    seed = 12

    size_threshold = 25
    stdev_threshold = 0.05
    name_length_threshold = 100
    
    train_fns_file = '../../../data/train_fn_ids.txt'
    test_fns_file = '../../../data/test_fn_ids.txt'
    fn_names_to_divide_file = '../../../data/fn_names_to_divide.txt'
    config_file = None

    configs = locals()

    return configs
    
    
    
    
