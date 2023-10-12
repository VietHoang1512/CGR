#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=47:59:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100GB
#SBATCH --job-name=celebA
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate /scratch/hvp2011/envs/python310
nvidia-smi

cd /scratch/hvp2011/implement/dfr/dfr_group_DRO/


# python run_expt.py --log_dir logs/celebA/prompt/imtl/sgd/scheduler/seed=0 --seed 0  --moo_method imtl  --num_tokens 10  --lr .1 --batch_size 128 --weight_decay 0.0001 --model imagenet21k_ViT-B_16 --warmup 10 --n_epochs 100 --log_every 200 --reweight_groups    -s confounder -d CelebA  --root_dir /scratch/hvp2011/implement/spurious-correlation/data/celebA  -t Blond_Hair -c Male
# python run_expt.py --log_dir logs/celebA/prompt/pcgrad/sgd/scheduler/seed=0 --seed 0  --moo_method pcgrad  --num_tokens 10  --lr .1 --batch_size 128 --weight_decay 0.0001 --model imagenet21k_ViT-B_16 --warmup 10 --n_epochs 100 --log_every 200 --reweight_groups    -s confounder -d CelebA  --root_dir /scratch/hvp2011/implement/spurious-correlation/data/celebA  -t Blond_Hair -c Male

python run_expt.py --seed 0  --moo_method epo --preference 1 1 1 1  --scheduler --num_tokens 10  --lr .1 --batch_size 128 --alpha 2 --weight_decay 0.0001 --model imagenet21k_ViT-B_16 --warmup 10 --n_epochs 50 --log_every 200 --data_dir /scratch/hvp2011/implement/spurious-correlation/data/celebA


# python run_expt.py --seed 1  --moo_method epo --preference 1 1 1 1  --scheduler --num_tokens 10  --lr .1 --batch_size 128 --alpha 1 --weight_decay 0.0001 --model imagenet21k_ViT-B_16 --warmup 10 --n_epochs 50 --log_every 200 --reweight_groups
# python run_expt.py --seed 2  --moo_method epo --preference 1 1 1 1  --scheduler --num_tokens 10  --lr .1 --batch_size 128 --alpha 1 --weight_decay 0.0001 --model imagenet21k_ViT-B_16 --warmup 10 --n_epochs 50 --log_every 200 --reweight_groups
