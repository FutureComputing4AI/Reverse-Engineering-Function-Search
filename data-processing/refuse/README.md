**assemblage**  
Given an Assemblage dataset and accompanying *.sqlite* database of metadata, filter and 
divide the functions into a training set and testing set. Save the function ids for each 
set to a file. Also, get a list of function names that indicate when functions should be
labeled not just by their names, but also by the originating source code.

For more information about how we constructed ML datasets from Assemblage, please refer to Section 4.1 of our paper: [link](ADD LINK).

1. Edit *dataset_configs.py* to set the dataset creation configurations.
2. Run *process_data.py*.

**ghidra**  
Use Ghidra to extract function bytes from binaries. 
Extracts each function into its own *.bin* file. File naming scheme is: *binarySHA256Hash\functionName\startRVA\endRVA\bytes.bin**.
This was the process used to test REFuSe on the MOTIF, BinaryCorp, and Marcelli Dataset-1 datasets.

1. Follow the instructions at the link below to build a *ghidra.jar* file:  
https://github.com/NationalSecurityAgency/ghidra/blob/master/Ghidra/RuntimeScripts/Common/support/buildGhidraJarREADME.txt
2. Edit line 3 in *run_ghidra.sh* to set the path to the directory of binaries to process.
3. Edit line 13 in *GhidraExtractFunctionBytes.java* to set the path to the output directory.
4. Run *./run_ghidra.sh*.
