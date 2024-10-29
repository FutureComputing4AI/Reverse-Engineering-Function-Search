Use a saved REFuSe checkpoint to generate function embeddings.

0. Make sure the data is processed. 

For basic datasets built with Assemblage, like the 
Common Libraries dataset, you need the path to the *sqlite* database file and the path to 
the binaries themselves. The basic Assemblage datasets uses our *AssemblageFunctionsDatasetBasic*
class, and will include every function in the sqlite table, each labeled according to its name. 
To select only specific function IDs, downsample singletons, or use a different labeling scheme, 
we also provide a broader *AssemblageFunctionsDataset* class. The code under
*refuse-public/data-processing/refuse/assemblage* can help to preprocess Assemblage datasets
and get some of the extra features.

For a dataset of non-Assemblage binaries, the bytes of each function should be extracted into their own files.
Code to do this with Ghidra can be found at *refuse-public/data-processing/refuse*.

1. Update the settings in *utils/gen_embeddings_configs.py*.

2. Run *python3 generate_embeddings.py* to generate the embeddings.