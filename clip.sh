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


# python run_expt.py --seed 0 --moo_method epo --preference 1 1 1 1  --log_every 50   --num_tokens 16  --lr .03 --scheduler --warmup 5  --batch_size 64 --weight_decay 0.001 --alpha 2.0 --model clip-RN50  --n_epochs 100 --data_transform WaterbirdsForCLIPTransform --data_dir /vast/hvp2011/data/waterbird_complete95_forest2water2/
python run_expt.py --seed 0 --moo_method epo --preference 1 1 1 1  --log_every 50   --num_tokens 16  --lr .003 --batch_size 64 --weight_decay 0.001 --alpha 1.5 --model clip-RN50  --n_epochs 100 --data_transform WaterbirdsForCLIPTransform --data_dir /vast/hvp2011/data/waterbird_complete95_forest2water2/
