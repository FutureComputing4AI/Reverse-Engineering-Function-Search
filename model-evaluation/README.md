Evaluate REFuSe, jTrans, and the GNN on our datasets. 

**Environment**
Run *conda env create --file refuse-eval.yaml* to create a conda environment with the necessary requirements. If desired, you can use mamba instead of conda.

**REFuSe / Transformer / GNN**
1. Follow the instructions in the *refuse/transformer/gnn* directories to generate embeddings over the
desired dataset using a trained model. 

2. Modify the configurations in *evaluation_configs.py*.

3. Run *python3 evaluate_embeddings.py* to evaluate the embeddings.

**jTrans**  
We were unable to modify the jTrans codebase to only extract function embeddings, so we rely
on the jTrans authors' original code to do evaluation as well. Follow the instructions in 
the jTrans directory to evaluate jTrans on our datasets.

**Note on MOTIF:** Ghidra/IDA Pro may struggle to accurately analyze the disarmed MOTIF 
binaries available on [GitHub](https://github.com/boozallen/MOTIF). The original (live) 
malware can be obtained from VirusTotal by hash.  
