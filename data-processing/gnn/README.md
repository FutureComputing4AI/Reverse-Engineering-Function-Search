Code for preprocessing data and running the GNN is made available on GitHub as part of the 
survey paper from Marcelli et. al, *How Machine Learning Is Solving the Binary Function 
Similarity Problem.* The accompanying code can be found 
[here](https://github.com/Cisco-Talos/binary_function_similarity/tree/main/Models/GGSNN-GMN), 
and comes with a descriptive README. Our code (and documentation) is intended to complement, not
replace, that work. Therefore, we only provide and discuss the code needed to extend this
work to our datasets. Readers should also familiarize themselves with the original repository 
before running our experiments.

**Note: running this code will require a copy of IDA Pro.** We use IDA Pro 8.2.

# Preprocessing Code for the GNN

0. Clone the Marcelli GitHub repository: *git clone https://github.com/Cisco-Talos/binary_function_similarity.git*

1. Integrate our supplemental code:

	1. Replace *binary_function_similarity/IDA_scripts/generate_idbs.py* with 
           *refuse-public/data-processing/gnn/generate_idbs.py*, which includes support for 
           the MOTIF, CommonLibraries, BinaryCorp, and Assemblage datasets. Update the paths to 
           those dataset files in lines 177, 182, 187, 192, and 197.
	
	2. Replace *binary_function_similarity/IDA_scripts/IDA_flowchart/IDA_flowchart.py* with
	   *refuse-public/data-processing/gnn/IDA_flowchart.py*. Our changes include:
	      - We do not skip functions with fewer than 5 basic blocks
	      - We record the start and stop *relative* virtual address, instead of the virtual 
	        address, for each function. This allows us to align the IDA output with the
	        metadata accompanying the Assemblage dataset.

	3. You may also need to replace 
	   *binary_function_similarity/IDA_scripts/IDA_acfg_disasm/IDA_acfg_disasm.py* with
	   *refuse-public/data-processing/gnn/IDA_acfg_disasm.py*. The original Marcelli code is
	   written for IDA Pro 7.3, and some syntax modifications are needed for compatibility
	   with newer versions. Our code is known to work with IDA Pro 8.2.
   
   Preprocessing may be slow on large datasets. In *refuse-public/data-processing/gnn/parallel*
   we provide code that can be used to parallelize many of the original Marcelli scripts. These
   scripts can be used in place of the Marcelli and/or *refuse-public* scripts of the same name,
   with the following caveats:

	- All scripts default to using all available cores.
	- *refuse-public/data-processing/gnn/parallel/generate_idbs.py* assumes all the input 
	  files are in the same folder (flat directory structure).
	- *refuse-public/data-processing/gnn/parallel/cli_flowchart.py* produces NUM\_CORES csv
	  output files. These files must then be combined into a single 
	  *flowchart_Dataset-dataset.csv* file. This can be done from the command line.
 
2. Preprocess the data using the code provided by Marcelli, after making the modifications
   indicated by Steps 1.1 and 1.2. **This will require IDA Pro**. Refer to the Marcelli 
   [README](https://github.com/Cisco-Talos/binary_function_similarity/tree/main/IDA_scripts).
   Our instructions below, supplement, but do not replace, the ones included there. 
      
   1. Follow the Marcelli README instructions to set up the needed requirements.
				
   2. Set the appropriate file paths in *generateidbs.py*. Then run *python3 
      generateidbs.py --dataset*, where "dataset" is either "motif", 
      "commonlibraries", "binarycorp", "assemblagetest", or "assemblagetrain".
	  When possible, we recommend making the *.pdb* files for a dataset
      available to IDA. This will allow for more accurate analysis. 
	   
   3. Follow the Marcelli README instructions to run the IDA Flowchart plugin over the IDBs.
	
   4. Run *python3 refuse-public/data-processing/gnn/Dataset-dataset creation.py*, where
	  "dataset" is "MOTIF", "CommonLibraries", "BinaryCorp", or "AssemblageSplit". Set the
	  following paths:
	   
	  - All Datasets: modify line #44 (output directory) and line #48 (path to *.csv* 
	    file generated in Step 2.3). 
	  - MOTIF: specify line #49, the path to the *motif_dataset.jsonl* file. 
	    (This can be download at https://github.com/boozallen/MOTIF).
	  - Assemblage: specify lines #47, #49, #50, and #51 to set the Assemblage SPLIT
	    name (Train or Test), path to a text file containing the SPLIT function IDs (as
	    generated by *refuse-public/data-processing/refuse/assemblage/process_data.py*),
	    path to the Assemblage sqlite database, and path to the Assemblage SPLIT IDBs. 
	   
	   Place the resulting *Dataset-dataset.csv* file in *binary_function_similarity/DBs/Dataset-dataset*,
	   where "dataset" is "MOTIF", "CommonLibraries", "BinaryCorp", or "AssemblageSplit".
	   
	5.  Follow the Marcelli README instructions to run the IDA ACFG_disasm plugin over the *.json*
	    generated in Step 2.4.
	    
	6. Navigate to *binary_function_similarity/Models/GGSNN-GMN*. Follow the instructions in
	   Part 1 of the Marcelli README to run *gnn_preprocessing.py* in *--training* mode, 
	   passing as input the output from Step 2.5. Set the output directory as 
	   *binary_function_similarity/Models/GGSNN-GMN/Preprocessing/Dataset-dataset*, where
	   "dataset" is "MOTIF", "CommonLibraries", "BinaryCorp", "AssemblageTrain", or "AssemblageTest".