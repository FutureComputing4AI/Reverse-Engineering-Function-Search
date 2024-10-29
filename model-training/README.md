Train models on Assemblage data.

**REFuSe**
1. Preprocess the Assemblage data following the instructions in 
   *REFUSE-public/data-processing/refuse/README.md*.
   
2. Run *conda env create --file refuse-training.yaml* to create a conda environment with the necessary requirements. 
   If desired, you can use mamba instead of conda. Note: this environment was designed to work with CUDA 12.2.

3. Edit *utils/configs.py* to set the training run configurations.

4. Run *python3 train.py*

**GNN**  
1. Preprocess the Assemblage data following the instructions in 
   *REFUSE-public/data-processing/refuse/README.md* to create a train/test split.
   
2. Further preprocess the Assemblage data for the GNN following the instructions in
   *REFUSE-public/data-processing/gnn/README.md*. Adhere to the specifications for 
   processing the Assemblage data, with SPLIT=Train.
   
3. Clone the Marcelli GitHub repository: *git clone https://github.com/Cisco-Talos/binary_function_similarity.git*

4. Integrate our supplemental code:
       	          
    1. Replace *binary_function_similarity/Models/GGSNN-GMN/NeuralNetwork/gnn.py* with 
       *refuse-public/model-training/gnn/gnn.py*. Our version adds support for training
       on the Assemblage dataset.
       
    2. Replace *binary_function_similarity/Models/GGSNN-GMN/NeuralNetwork/core/config.py* 
       with *refuse-public/model-training/gnn/config.py*. This file adds support for 
       training on the Assemblage dataset.

    3. Replace *binary_function_similarity/Models/GGSNN-GMN/NeuralNetwork/core/build_dataset.py*
       with *refuse-public/model-training/gnn/build_dataset.py*. This file allows the training
       and validation set dataloaders to be generated separately.

    4. Replace *binary_function_similarity/Models/GGSNN-GMN/NeuralNetwork/core/gnn_model.py*
       with *refuse-public/model-training/gnn/gnn_model.py*. This file adapts the original 
       Marcelli code to support training on the Assemblage dataset.
       
5. Follow the instructions in Part 2 of the README in *binary_function_similarity/Models/GGSNN-GMN*. 
   After building the docker image, run the following docker command:
   
   ```
   docker run --rm \
    -v $(pwd)/../../DBs:/input \
    -v $(pwd)/NeuralNetwork:/output \
    -v $(pwd)/Preprocessing:/preprocessing \
    -it gnn-neuralnetwork /code/gnn.py --train --num_epochs 10 \
        --model_type embedding --training_mode pair \
        --features_type opc --dataset assemblagetrain \
        -c /output/model_checkpoint_$(date +'%Y-%m-%d') \
        -o /output/Dataset-AssemblageTrain
   ``` 
   
   The new model checkpoint will be at 
   *binary_function_similarity/Models/GGSNN-GMN/model_checkpoint_YYYY-MM-DD*.
   
**jTrans**  
Follow the instructions in the README in the jTrans directory. Please note that the code for 
pre-training jTrans is not publicly available. Our code uses the released pre-trained jTrans 
model and fine-tunes it on Assemblage data.

**Note:** We were only able to fine-tune jTrans on a subset of the Assemblage training dataset
        (25M functions) due to compute limitations.


**Naive Transformer Encoder**

The Transformer Encoder uses the same data pre-processing and conda environment used to train REFuSe, and most of the non-model configurations are the same with the exception of the reduced batch size to preserve memory. 

Like REFuSe, the training is configured in *utils/configs.py* and initiated with *python3 train.py*

## Environment and Data Setup

Training and evaluation for the Transformer Encoder use the same respective conda environments and pre-processing as REFuSe.

## Configuration

Training Parameters are set in `model-training/transformer/utils/configs.py`

Model Hyperparameters are set in the `TransformerConfig` dataclass of `model-training/transformer/utils/net_modules.py`