#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=47:59:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=100GB
#SBATCH --job-name=isic
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate /scratch/hvp2011/envs/python310
nvidia-smi

cd /scratch/hvp2011/implement/dfr/dfr_group_DRO/


# alpha=2.
# lr=.0003

# for weight_decay in  .01 .001 
# do
#     for num_tokens in 5 10
#     do
#         python run_expt.py  --seed 0  --moo_method epo --preference 1 1 1 1 --log_every 30   --num_tokens $num_tokens  --lr $lr --batch_size 16 --weight_decay $weight_decay --alpha $alpha --model imagenet21k_ViT-B_16  --n_epochs 50 --dataset ISICDataset --data_transform ISICTransform --data_dir /vast/hvp2011/data/isic
#         python run_expt.py  --seed 1  --moo_method epo --preference 1 1 1 1 --log_every 30   --num_tokens $num_tokens  --lr $lr --batch_size 16 --weight_decay $weight_decay --alpha $alpha --model imagenet21k_ViT-B_16  --n_epochs 50 --dataset ISICDataset --data_transform ISICTransform --data_dir /vast/hvp2011/data/isic
#         python run_expt.py  --seed 2  --moo_method epo --preference 1 1 1 1 --log_every 30   --num_tokens $num_tokens  --lr $lr --batch_size 16 --weight_decay $weight_decay --alpha $alpha --model imagenet21k_ViT-B_16  --n_epochs 50 --dataset ISICDataset --data_transform ISICTransform --data_dir /vast/hvp2011/data/isic
#     done
# done


# python run_expt.py  --seed 1  --moo_method epo --preference 1 1 1 1 --log_every 30   --num_tokens 10  --lr .003 --batch_size 16 --weight_decay .001 --alpha 1.5 --model imagenet21k_ViT-B_16  --n_epochs 50 --dataset ISICDataset --data_transform ISICTransform --data_dir /vast/hvp2011/data/isic


python run_expt.py --seed 1215 --moo_method epo --preference 1 1 1 1 --log_every 30 --num_tokens 5 --lr .003 --batch_size 16 --weight_decay .01 --alpha 1.5 --model imagenet21k_ViT-B_16 --n_epochs 50 --dataset ISICDataset --data_transform ISICTransform --data_dir /vast/hvp2011/data/isic
