python run_expt.py -s confounder -d CUB -t waterbird_complete95 --log_dir logs/wb-old/ --root_dir /scratch/hvp2011/implement/spurious-correlation/data/waterbird_complete95_forest2water2 -c forest2water2 --lr 0.001 --batch_size 128 --weight_decay 0.0001 --model imagenet_resnet50_pretrained --n_epochs 300 --reweight_groups --robust --gamma 0.1 --generalization_adjustment 0