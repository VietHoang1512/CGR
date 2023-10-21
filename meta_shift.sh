#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=5:59:00
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

# alpha=2
# lr=.03
# wd=.1

# for num_tokens in 5 10
# do
#     python run_expt.py  --seed 0  --moo_method epo --preference 1 1 1 1 --log_every 10   --num_tokens $num_tokens  --lr $lr --batch_size 16 --weight_decay $wd --alpha $alpha  --model imagenet21k_ViT-B_16  --n_epochs 50 --dataset MetaShiftDataset --data_dir /vast/hvp2011/data/metashifts/MetaDatasetCatDog 
#     python run_expt.py  --seed 1  --moo_method epo --preference 1 1 1 1 --log_every 10   --num_tokens $num_tokens  --lr $lr --batch_size 16 --weight_decay $wd --alpha $alpha  --model imagenet21k_ViT-B_16  --n_epochs 50 --dataset MetaShiftDataset --data_dir /vast/hvp2011/data/metashifts/MetaDatasetCatDog 
#     python run_expt.py  --seed 2  --moo_method epo --preference 1 1 1 1 --log_every 10   --num_tokens $num_tokens  --lr $lr --batch_size 16 --weight_decay $wd --alpha $alpha  --model imagenet21k_ViT-B_16  --n_epochs 50 --dataset MetaShiftDataset --data_dir /vast/hvp2011/data/metashifts/MetaDatasetCatDog 
# done


# for lr in 0.01 0.003 0.03 
# do
#     for wd in .1 .01 .001 
#     do
#         for num_tokens in 5 10
#         do
#             python run_expt.py  --seed 0  --moo_method epo --preference 1 1 1 1 --log_every 10   --num_tokens $num_tokens  --lr $lr --batch_size 16 --weight_decay $wd --alpha $alpha  --model imagenet21k_ViT-B_16  --n_epochs 50 --dataset MetaShiftDataset --data_dir /vast/hvp2011/data/metashifts/MetaDatasetCatDog 
#             python run_expt.py  --seed 1  --moo_method epo --preference 1 1 1 1 --log_every 10   --num_tokens $num_tokens  --lr $lr --batch_size 16 --weight_decay $wd --alpha $alpha  --model imagenet21k_ViT-B_16  --n_epochs 50 --dataset MetaShiftDataset --data_dir /vast/hvp2011/data/metashifts/MetaDatasetCatDog 
#             python run_expt.py  --seed 2  --moo_method epo --preference 1 1 1 1 --log_every 10   --num_tokens $num_tokens  --lr $lr --batch_size 16 --weight_decay $wd --alpha $alpha  --model imagenet21k_ViT-B_16  --n_epochs 50 --dataset MetaShiftDataset --data_dir /vast/hvp2011/data/metashifts/MetaDatasetCatDog 
#         done
#     done
# done

# python run_expt.py  --seed 0  --moo_method epo --preference 1 1 1 1 --log_every 10   --lr .001 --batch_size 16 --weight_decay .0001 --alpha .8  --model imagenet_resnet50_pretrained  --n_epochs 50 --dataset MetaShiftDataset --data_dir /vast/hvp2011/data/metashifts/MetaDatasetCatDog


# for lr in 0.003 0.001 0.0003 0.0001 
# do
#     for alpha in  1 .8 .5 1.2 1.5 2
#     do
#         for wd in .1 .01 .001 
#         do

#                 python run_expt.py  --seed 0  --moo_method epo --preference 1 1 1 1 --log_every 10    --lr $lr --batch_size 16 --weight_decay $wd --alpha $alpha  --model imagenet_resnet50_pretrained  --n_epochs 50 --dataset MetaShiftDataset --data_dir /vast/hvp2011/data/metashifts/MetaDatasetCatDog 
#                 python run_expt.py  --seed 1  --moo_method epo --preference 1 1 1 1 --log_every 10    --lr $lr --batch_size 16 --weight_decay $wd --alpha $alpha  --model imagenet_resnet50_pretrained  --n_epochs 50 --dataset MetaShiftDataset --data_dir /vast/hvp2011/data/metashifts/MetaDatasetCatDog 
#                 python run_expt.py  --seed 2  --moo_method epo --preference 1 1 1 1 --log_every 10    --lr $lr --batch_size 16 --weight_decay $wd --alpha $alpha  --model imagenet_resnet50_pretrained  --n_epochs 50 --dataset MetaShiftDataset --data_dir /vast/hvp2011/data/metashifts/MetaDatasetCatDog 
#         done    
#     done
# done


lr=0.0001
alpha=2.
for wd in .1 .01 .001 
do
    # python run_expt.py  --seed 0  --moo_method epo --preference 1 1 1 1 --log_every 10    --lr $lr --batch_size 16 --weight_decay $wd --alpha $alpha  --model imagenet_resnet50_pretrained  --n_epochs 50 --dataset MetaShiftDataset --data_dir /vast/hvp2011/data/metashifts/MetaDatasetCatDog 
    # python run_expt.py  --seed 1  --moo_method epo --preference 1 1 1 1 --log_every 10    --lr $lr --batch_size 16 --weight_decay $wd --alpha $alpha  --model imagenet_resnet50_pretrained  --n_epochs 50 --dataset MetaShiftDataset --data_dir /vast/hvp2011/data/metashifts/MetaDatasetCatDog 
    python run_expt.py  --seed 2  --moo_method epo --preference 1 1 1 1 --log_every 10    --lr $lr --batch_size 16 --weight_decay $wd --alpha $alpha  --model imagenet_resnet50_pretrained  --n_epochs 50 --dataset MetaShiftDataset --data_dir /vast/hvp2011/data/metashifts/MetaDatasetCatDog
done    