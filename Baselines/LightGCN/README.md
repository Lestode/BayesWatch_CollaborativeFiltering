# LightGCN

This model can be trained on the slurm cluster.
For that :
1. upload the files ```LightGCN_job.sh``` and ```LightGCN.py``` to the cluster.
2. Create two directories on the cluster: ```logs``` and ```lightgcn_output```
3. Enter the desired hyper-parameters in ```LightGCN_job.sh```
4. Run the commands ```module load cuda/12.6.0``` and ```conda activate /cluster/courses/cil/envs/collaborative_filtering/```
5. Run the script using ```sbatch LightGCN.sh```

# Requirements

- Python 3.8 or newer  
- pandas  
- numpy  
- PyTorch  
- scikit-learn
- PyTorch Geometric