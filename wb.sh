#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=47:59:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=100GB
#SBATCH --job-name=wb
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate /scratch/hvp2011/envs/python310
nvidia-smi

cd /scratch/hvp2011/implement/dfr/dfr_group_DRO/

python run_expt.py --log_dir logs/wb/resnet/sgd/pcgrad/seed=0 --seed 0  --moo_method pcgrad  --lr 0.001 --batch_size 128 --weight_decay 0.0001 --model imagenet_resnet50_pretrained --n_epochs 200 --reweight_groups   --root_dir /scratch/hvp2011/implement/spurious-correlation/data/waterbird_complete95_forest2water2  -s confounder -d CUB -t waterbird_complete95 -c forest2water2
python run_expt.py --log_dir logs/wb/resnet/sgd/pcgrad/seed=1 --seed 1  --moo_method pcgrad  --lr 0.001 --batch_size 128 --weight_decay 0.0001 --model imagenet_resnet50_pretrained --n_epochs 200 --reweight_groups   --root_dir /scratch/hvp2011/implement/spurious-correlation/data/waterbird_complete95_forest2water2  -s confounder -d CUB -t waterbird_complete95 -c forest2water2
python run_expt.py --log_dir logs/wb/resnet/sgd/pcgrad/seed=2 --seed 2  --moo_method pcgrad  --lr 0.001 --batch_size 128 --weight_decay 0.0001 --model imagenet_resnet50_pretrained --n_epochs 200 --reweight_groups   --root_dir /scratch/hvp2011/implement/spurious-correlation/data/waterbird_complete95_forest2water2  -s confounder -d CUB -t waterbird_complete95 -c forest2water2

# python run_expt.py --log_dir logs/wb-prompt/imtl/sgd/seed=0 --seed 0  --moo_method imtl  --num_tokens 3  --lr 3e-5 --batch_size 128 --weight_decay 0.0001 --model imagenet21k_ViT-B_16 --n_epochs 200 --reweight_groups     --root_dir /scratch/hvp2011/implement/spurious-correlation/data/waterbird_complete95_forest2water2  -s confounder -d CUB -t waterbird_complete95 -c forest2water2


