#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=100GB
#SBATCH --job-name=clip
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate /scratch/hvp2011/envs/python310
nvidia-smi

cd /scratch/hvp2011/implement/dfr/dfr_group_DRO/



alpha=1.5
lr=0.01
weight_decay=0.01
batch_size=32
# for seed in 0 42 2023
# do
#     for num_tokens in 8 16 
#     do
#         python run_expt.py --seed $seed --moo_method epo --preference 1 1 1 1  --log_every 50   --num_tokens $num_tokens  --lr $lr --scheduler --warmup 5  --batch_size $batch_size  --weight_decay $weight_decay --alpha $alpha --model clip-RN50  --n_epochs 100 --data_transform WaterbirdsForCLIPTransform --data_dir /vast/hvp2011/data/waterbird_complete95_forest2water2/
#         python run_expt.py --seed $seed --moo_method epo --preference 1 1 1 1  --log_every 50   --num_tokens $num_tokens  --lr $lr --batch_size $batch_size  --weight_decay $weight_decay --alpha $alpha --model clip-RN50  --n_epochs 100 --data_transform WaterbirdsForCLIPTransform --data_dir /vast/hvp2011/data/waterbird_complete95_forest2water2/
#     done
# done
# python run_expt.py --seed 0 --moo_method epo --preference 1 1 1 1  --log_every 50   --num_tokens $num_tokens  --lr $lr --scheduler --warmup 5  --batch_size $batch_size  --weight_decay $weight_decay --alpha $alpha --model clip-ViT-L/14@336px  --n_epochs 100 --data_transform WaterbirdsForBigCLIPTransform --data_dir /vast/hvp2011/data/waterbird_complete95_forest2water2/
# python run_expt.py --seed 42 --moo_method epo --preference 1 1 1 1  --log_every 50   --num_tokens $num_tokens  --lr $lr --scheduler --warmup 5  --batch_size $batch_size  --weight_decay $weight_decay --alpha $alpha --model clip-ViT-L/14@336px  --n_epochs 100 --data_transform WaterbirdsForBigCLIPTransform --data_dir /vast/hvp2011/data/waterbird_complete95_forest2water2/
# python run_expt.py --seed 2023 --moo_method epo --preference 1 1 1 1  --log_every 50   --num_tokens $num_tokens  --lr $lr --scheduler --warmup 5  --batch_size $batch_size  --weight_decay $weight_decay --alpha $alpha --model clip-ViT-L/14@336px  --n_epochs 100 --data_transform WaterbirdsForBigCLIPTransform --data_dir /vast/hvp2011/data/waterbird_complete95_forest2water2/

# python run_expt.py --seed 1512 --moo_method epo --preference 1 1 1 1  --log_every 50   --num_tokens $num_tokens  --lr $lr --scheduler --warmup 5  --batch_size $batch_size  --weight_decay $weight_decay --alpha $alpha --model clip-RN50  --n_epochs 100 --data_transform WaterbirdsForCLIPTransform --data_dir /vast/hvp2011/data/waterbird_complete95_forest2water2/
seed=0
num_tokens=16
CUDA_VISIBLE_DEVICES=0 python run_expt.py --seed $seed --moo_method epo --preference 1 1 1 1  --log_every 50   --num_tokens $num_tokens  --lr $lr --scheduler --warmup 5  --batch_size $batch_size  --weight_decay $weight_decay --alpha $alpha --model clip-ViT-L/14@336px  --n_epochs 100 --data_transform WaterbirdsForBigCLIPTransform --data_dir /vast/hvp2011/data/waterbird_complete95_forest2water2/ &
CUDA_VISIBLE_DEVICES=1 python run_expt.py --seed $seed --moo_method epo --preference 1 1 1 1  --log_every 50   --num_tokens $num_tokens  --lr $lr --batch_size $batch_size  --weight_decay $weight_decay --alpha $alpha --model clip-ViT-L/14@336px  --n_epochs 100 --data_transform WaterbirdsForBigCLIPTransform --data_dir /vast/hvp2011/data/waterbird_complete95_forest2water2/ &
wait