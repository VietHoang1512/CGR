python run_expt.py -s confounder -d CelebA  --root_dir /scratch/hvp2011/implement/spurious-correlation/data/celebA -t Blond_Hair -c Male --lr 0.0001 --batch_size 128 --weight_decay 0.0001 --model imagenet_resnet50_pretrained --n_epochs 50 --reweight_groups --robust --gamma 0.1 --generalization_adjustment 0 --log_dir logs/celebA