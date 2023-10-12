#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
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


# python run_expt.py --seed 2 --moo_method epo  --preference 1 1 1 1 --lr .001 --batch_size 128 --weight_decay 0.001 --alpha 2 --model imagenet_resnet50_pretrained --warmup 0 --n_epochs 100   --data_dir /scratch/hvp2011/implement/spurious-correlation/data/waterbird_complete95_forest2water2    
# python run_expt.py --seed 2 --moo_method imtl   --lr .001 --batch_size 128 --weight_decay 0.0001 --model imagenet_resnet50_pretrained --warmup 0 --n_epochs 100   --data_dir /scratch/hvp2011/implement/spurious-correlation/data/waterbird_complete95_forest2water2    
# python run_expt.py --seed 2 --moo_method ew   --lr .001 --batch_size 128 --weight_decay 0.0001 --model imagenet_resnet50_pretrained --warmup 0 --n_epochs 100   --data_dir /scratch/hvp2011/implement/spurious-correlation/data/waterbird_complete95_forest2water2    

# python run_expt.py --seed 0  --moo_method ew   --num_tokens 5  --lr .2 --batch_size 128 --weight_decay 0.0001 --model imagenet21k_ViT-B_16  --n_epochs 100 --data_dir /scratch/hvp2011/implement/spurious-correlation/data/waterbird_complete95_forest2water2 
# python run_expt.py  --seed 0  --moo_method epo --preference 1 1 1 1 --log_every 50   --num_tokens 20  --lr .1 --scheduler --batch_size 64 --weight_decay 0.001 --alpha 2 --model imagenet21k_ViT-B_16  --n_epochs 100 --data_dir /scratch/hvp2011/implement/spurious-correlation/data/waterbird_complete95_forest2water2


# python run_expt.py  --seed 10  --moo_method epo --preference 1 1 1 1 --log_every 50   --num_tokens 10  --lr .1 --batch_size 64 --weight_decay 0.001 --alpha 2 --model imagenet21k_ViT-B_16  --n_epochs 100 --data_dir /scratch/hvp2011/implement/spurious-correlation/data/waterbird_complete95_forest2water2 
python run_expt.py  --seed 1215  --moo_method epo --preference 1 1 1 1 --log_every 50   --num_tokens 10  --lr .1 --batch_size 64 --weight_decay 0.001 --alpha 2 --model imagenet21k_ViT-B_16  --n_epochs 100 --data_dir /scratch/hvp2011/implement/spurious-correlation/data/waterbird_complete95_forest2water2 
# python run_expt.py  --seed 2000  --moo_method epo --preference 1 1 1 1 --log_every 50   --num_tokens 10  --lr .1 --batch_size 64 --weight_decay 0.001 --alpha 2 --model imagenet21k_ViT-B_16  --n_epochs 100 --data_dir /scratch/hvp2011/implement/spurious-correlation/data/waterbird_complete95_forest2water2 

# python run_expt.py  --seed 1  --moo_method epo --preference 1 1 1 1 --log_every 50   --num_tokens 10  --lr .1 --batch_size 64 --weight_decay 0.001 --alpha 2 --model imagenet21k_ViT-B_16  --n_epochs 100 --data_dir /scratch/hvp2011/implement/spurious-correlation/data/waterbird_complete95_forest2water2 
# python run_expt.py  --seed 2  --moo_method epo --preference 1 1 1 1 --log_every 50   --num_tokens 10  --lr .1 --batch_size 64 --weight_decay 0.001 --alpha 2 --model imagenet21k_ViT-B_16  --n_epochs 100 --data_dir /scratch/hvp2011/implement/spurious-correlation/data/waterbird_complete95_forest2water2 

# python run_expt.py  --seed 0  --moo_method epo --preference 1 1 1 1 --log_every 50   --num_tokens 10  --lr .0001 --batch_size 64 --weight_decay 0.0001 --alpha 1 --model imagenet_resnet50_pretrained  --n_epochs 100 --data_dir /scratch/hvp2011/implement/spurious-correlation/data/waterbird_complete95_forest2water2 
# python run_expt.py  --seed 1  --moo_method epo --preference 1 1 1 1 --log_every 50   --num_tokens 10  --lr .001 --batch_size 64 --weight_decay 0.0001 --alpha 1 --model imagenet_resnet50_pretrained  --n_epochs 100 --data_dir /scratch/hvp2011/implement/spurious-correlation/data/waterbird_complete95_forest2water2 
# python run_expt.py  --seed 2  --moo_method epo --preference 1 1 1 1 --log_every 50   --num_tokens 10  --lr .001 --batch_size 64 --weight_decay 0.0001 --alpha 1 --model imagenet_resnet50_pretrained  --n_epochs 100 --data_dir /scratch/hvp2011/implement/spurious-correlation/data/waterbird_complete95_forest2water2 

