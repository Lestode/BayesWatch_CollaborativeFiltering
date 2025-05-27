# LightGCN

This model can be trained on the slurm cluster.

1. upload the svd_embedding directory adn its content to the cluster, or run the ```SVDpp_embeddings.ipynb``` notebook to generate new svd embeddings.
2. upload the files ```run_wandb_sweep.sh``` and ```model.py``` to the cluster.
3. Create two directories on the cluster: ```logs``` and ```model_output```.
4. Create a WandB account and change the "project" and "entity" field in the ```wandb_sweep.yaml```.
5. Enter the desired hyper-parameters in ```wandb_sweep.yaml```.
6. Run the commands ```module load cuda/12.6.0``` and ```conda activate /cluster/courses/cil/envs/collaborative_filtering/```
7. run the command ```wandb sweep wandb_sweep.yaml```
8. copy the generated code into the ```run_wandb_sweep.sh``` file, where the there is the following signs: ***
9. Run using ```sbatch run_wandb_sweep.sh```

# Requirements

- Python 3.8 or newer  
- pandas  
- numpy  
- PyTorch  
- scikit-learn
- PyTorch Geometric
- wandb