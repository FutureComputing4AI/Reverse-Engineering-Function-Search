jTrans is made available on GitHub here: *https://github.com/vul337/jTrans/tree/main*
and comes with a descriptive README. Our code (and documentation) is intended to complement, not
replace, that work. Therefore, we only provide and discuss our supplemental code for extending
jTrans to Assemblage data. Readers should also familiarize themselves with the original jTrans
repository before running experiments with jTrans.

# Assemblage Adoption Code for jTrans

This folder holds the code necessary to process the Assemblage data (typically an sqlite database 
and a folder containing binaries) to the jTrans training data format.  

Please first obtain the jTrans original author's code at *https://github.com/vul337/jTrans* and 
install required modules. In addition, you will need to install the Python module `pefile` 
to work with the Assemblage data.

This folder has 2 sub-folders; the first sub-folder is `data processing`, which contains code to 
process the Assemblage binaries into the pickle data format that jTrans requires. 

### Data Processing

1. Navigate to the `data processing` folder, then move the `util` folder to the `datautils/util` 
   folder in the original author's code.

2. Place the Assemblage database file and the Assemblage binaries folder into the `datautils`
   folder described in Step 1. 

3. Copy the `jTrans_Assemblage_adopt.ipynb` code to the `datautils` folder and 
   change the names of database file and the training data file in code block one.

4. Run all code blocks in `jTrans_Assemblage_adopt.ipynb` in order. It might take hours to days. 
   The processed data will be available in the folder `extract_selected`. 
   
5. Create a train/test split by dividing the data in `extract_selected` into two sub-folders.

### Fine-tuning

1. Complete the steps in the Data Processing section. 

2. Replace the `data.py` file in the original jTrans code (root directory) with the `data.py`
   file in the `training codes` folder, and replace the `playdata.py` file in the jTrans
   `datautils` directory with the one provided in `training codes`.

3. Download the pre-trained jTrans model, which is linked on the jTrans GitHub page.

4. After providing the path to the training and testing folders, use the `finetune.py` script 
   provided by the jTrans authors to finetune the model. 