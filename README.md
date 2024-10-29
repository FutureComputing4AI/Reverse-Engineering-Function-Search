This repository contains the code used to build the models and run the experiments presented in *Is Function Similarity Over-Engineered? Building a Benchmark*. The paper can be found here (LINK TO COME).

Our paper evaluates five models: [jTrans](https://arxiv.org/pdf/2205.12713.pdf), a [GNN from Li et. al](https://proceedings.mlr.press/v97/li19d/li19d.pdf), A Naive Multiheaded-Attention Transformer Encoder, [Ghidra's BSim Plugin](https://ghidra-sre.org), and REFuSe, a new model introduced in our paper. These models are assessed against five datasets: [Assemblage](https://arxiv.org/pdf/2405.03991), [MOTIF](https://github.com/boozallen/MOTIF), CommonLibraries, [Marcelli Dataset-1](https://github.com/Cisco-Talos/binary_function_similarity), and [BinaryCorp](https://github.com/vul337/jTrans/).

**data**  
The recipe for building our Assemblage dataset, and the code to run our BSim experiments. A guide to reproducing datasets from recipes can be found [here](https://assemblagedocs.readthedocs.io/en/latest/deployment_windows.html#optional-recover-dataset), and more details about BSim can be found in the corresponding folder.

**data-processing**  
Preprocess data for experiments.

**model-training**  
Train models on Assemblage data.

**model-evaluation**  
Evaluate models on datasets.
