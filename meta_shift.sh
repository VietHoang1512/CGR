#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=47:59:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=100GB
#SBATCH --job-name=metashift
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate /scratch/hvp2011/envs/python310
nvidia-smi

cd /scratch/hvp2011/implement/dfr/dfr_group_DRO/

for lr in 0.01 0.003 0.03
do
    for wd in .1 .01 .001 .003 .03
    do
        python run_expt.py  --seed 0  --moo_method epo --preference 1 1 1 1 --log_every 10   --num_tokens 10  --lr $lr --batch_size 16 --weight_decay $wd --alpha 1.2 --model imagenet21k_ViT-B_16  --n_epochs 50 --dataset MetaShiftDataset --data_dir /scratch/hvp2011/implement/spurious-correlation/data/metashifts/MetaDatasetCatDog
        python run_expt.py  --seed 1  --moo_method epo --preference 1 1 1 1 --log_every 10   --num_tokens 10  --lr $lr --batch_size 16 --weight_decay $wd --alpha 1.2 --model imagenet21k_ViT-B_16  --n_epochs 50 --dataset MetaShiftDataset --data_dir /scratch/hvp2011/implement/spurious-correlation/data/metashifts/MetaDatasetCatDog
        python run_expt.py  --seed 2  --moo_method epo --preference 1 1 1 1 --log_every 10   --num_tokens 10  --lr $lr --batch_size 16 --weight_decay $wd --alpha 1.2 --model imagenet21k_ViT-B_16  --n_epochs 50 --dataset MetaShiftDataset --data_dir /scratch/hvp2011/implement/spurious-correlation/data/metashifts/MetaDatasetCatDog
    done
done
