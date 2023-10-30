import matplotlib.pyplot as plt

import torch

# reproduced
# EPOCH
erm_ckpt = "outputs/data_dir=/vast/hvp2011/data/metashifts/MetaDatasetCatDog/test_data_dir=None/data_transform=AugWaterbirdsCelebATransform/dataset=MetaShiftDataset/alpha=0.0/model=imagenet_resnet50_pretrained/train_from_scratch=False/moo_method=erm/num_tokens=10/multi_prompt=False/n_epochs=50/batch_size=16/lr=0.001/scheduler=False/weight_decay=0.0001/warmup=5/max_grad_norm=1/preference=None/seed=0/log_every=100/save_step=10/save_best=False/save_last=False/output_dir=outputs/data_dir=/vast/hvp2011/data/metashifts/MetaDatasetCatDog/test_data_dir=None/data_transform=AugWaterbirdsCelebATransform/dataset=MetaShiftDataset/alpha=0.0/model=imagenet_resnet50_pretrained/train_from_scratch=False/moo_method=erm/num_tokens=10/multi_prompt=False/n_epochs=50/batch_size=16/lr=0.001/scheduler=False/weight_decay=0.0001/warmup=5/max_grad_norm=1/preference=None/seed=0/log_every=100/save_step=10/save_best=False/save_last=False/stats.pt"
groupdro_ckpt = "outputs/data_dir=/vast/hvp2011/data/metashifts/MetaDatasetCatDog/test_data_dir=None/data_transform=AugWaterbirdsCelebATransform/dataset=MetaShiftDataset/alpha=1.0/model=imagenet_resnet50_pretrained/train_from_scratch=False/moo_method=groupdro/num_tokens=10/multi_prompt=False/n_epochs=50/batch_size=16/lr=0.001/scheduler=False/weight_decay=0.0001/warmup=5/max_grad_norm=1/preference=None/seed=0/log_every=100/save_step=10/save_best=False/save_last=False/output_dir=outputs/data_dir=/vast/hvp2011/data/metashifts/MetaDatasetCatDog/test_data_dir=None/data_transform=AugWaterbirdsCelebATransform/dataset=MetaShiftDataset/alpha=1.0/model=imagenet_resnet50_pretrained/train_from_scratch=False/moo_method=groupdro/num_tokens=10/multi_prompt=False/n_epochs=50/batch_size=16/lr=0.001/scheduler=False/weight_decay=0.0001/warmup=5/max_grad_norm=1/preference=None/seed=0/log_every=100/save_step=10/save_best=False/save_last=False/stats.pt"
our_ckpt = "outputs/data_dir=/vast/hvp2011/data/metashifts/MetaDatasetCatDog/test_data_dir=None/data_transform=AugWaterbirdsCelebATransform/dataset=MetaShiftDataset/alpha=1.2/model=imagenet_resnet50_pretrained/train_from_scratch=False/moo_method=epo/num_tokens=10/multi_prompt=False/n_epochs=50/batch_size=16/lr=0.003/scheduler=False/weight_decay=0.01/warmup=5/max_grad_norm=1/preference=['1', '1', '1', '1']/seed=0/log_every=100/save_step=10/save_best=False/save_last=False/output_dir=outputs/data_dir=/vast/hvp2011/data/metashifts/MetaDatasetCatDog/test_data_dir=None/data_transform=AugWaterbirdsCelebATransform/dataset=MetaShiftDataset/alpha=1.2/model=imagenet_resnet50_pretrained/train_from_scratch=False/moo_method=epo/num_tokens=10/multi_prompt=False/n_epochs=50/batch_size=16/lr=0.003/scheduler=False/weight_decay=0.01/warmup=5/max_grad_norm=1/preference=['1', '1', '1', '1']/seed=0/log_every=100/save_step=10/save_best=False/save_last=False/stats.pt"

# ITERATION
# erm_ckpt = "/scratch/hvp2011/implement/dfr/dfr_group_DRO/outputs/data_dir=/vast/hvp2011/data/metashifts/MetaDatasetCatDog/test_data_dir=None/data_transform=AugWaterbirdsCelebATransform/dataset=MetaShiftDataset/alpha=0.0/model=imagenet_resnet50_pretrained/train_from_scratch=False/moo_method=erm/num_tokens=10/multi_prompt=False/n_epochs=50/batch_size=16/lr=0.001/scheduler=False/weight_decay=0.0001/warmup=5/max_grad_norm=1/preference=None/seed=0/log_every=1/save_step=10/save_best=False/save_last=False/output_dir=outputs/data_dir=/vast/hvp2011/data/metashifts/MetaDatasetCatDog/test_data_dir=None/data_transform=AugWaterbirdsCelebATransform/dataset=MetaShiftDataset/alpha=0.0/model=imagenet_resnet50_pretrained/train_from_scratch=False/moo_method=erm/num_tokens=10/multi_prompt=False/n_epochs=50/batch_size=16/lr=0.001/scheduler=False/weight_decay=0.0001/warmup=5/max_grad_norm=1/preference=None/seed=0/log_every=1/save_step=10/save_best=False/save_last=False/stats.pt"
# groupdro_ckpt = "/scratch/hvp2011/implement/dfr/dfr_group_DRO/outputs/data_dir=/vast/hvp2011/data/metashifts/MetaDatasetCatDog/test_data_dir=None/data_transform=AugWaterbirdsCelebATransform/dataset=MetaShiftDataset/alpha=1.0/model=imagenet_resnet50_pretrained/train_from_scratch=False/moo_method=groupdro/num_tokens=10/multi_prompt=False/n_epochs=50/batch_size=16/lr=0.001/scheduler=False/weight_decay=0.0001/warmup=5/max_grad_norm=1/preference=None/seed=0/log_every=1/save_step=10/save_best=False/save_last=False/output_dir=outputs/data_dir=/vast/hvp2011/data/metashifts/MetaDatasetCatDog/test_data_dir=None/data_transform=AugWaterbirdsCelebATransform/dataset=MetaShiftDataset/alpha=1.0/model=imagenet_resnet50_pretrained/train_from_scratch=False/moo_method=groupdro/num_tokens=10/multi_prompt=False/n_epochs=50/batch_size=16/lr=0.001/scheduler=False/weight_decay=0.0001/warmup=5/max_grad_norm=1/preference=None/seed=0/log_every=1/save_step=10/save_best=False/save_last=False/stats.pt"
# our_ckpt = "/scratch/hvp2011/implement/dfr/dfr_group_DRO/outputs/data_dir=/vast/hvp2011/data/metashifts/MetaDatasetCatDog/test_data_dir=None/data_transform=AugWaterbirdsCelebATransform/dataset=MetaShiftDataset/alpha=1.2/model=imagenet_resnet50_pretrained/train_from_scratch=False/moo_method=epo/num_tokens=10/multi_prompt=False/n_epochs=50/batch_size=16/lr=0.003/scheduler=False/weight_decay=0.01/warmup=5/max_grad_norm=1/preference=['1', '1', '1', '1']/seed=0/log_every=1/save_step=10/save_best=False/save_last=False/output_dir=outputs/data_dir=/vast/hvp2011/data/metashifts/MetaDatasetCatDog/test_data_dir=None/data_transform=AugWaterbirdsCelebATransform/dataset=MetaShiftDataset/alpha=1.2/model=imagenet_resnet50_pretrained/train_from_scratch=False/moo_method=epo/num_tokens=10/multi_prompt=False/n_epochs=50/batch_size=16/lr=0.003/scheduler=False/weight_decay=0.01/warmup=5/max_grad_norm=1/preference=['1', '1', '1', '1']/seed=0/log_every=1/save_step=10/save_best=False/save_last=False/stats.pt"

# 10 ITERATION
erm_ckpt = "outputs/data_dir=/vast/hvp2011/data/metashifts/MetaDatasetCatDog/test_data_dir=None/data_transform=AugWaterbirdsCelebATransform/dataset=MetaShiftDataset/alpha=0.0/model=imagenet_resnet50_pretrained/train_from_scratch=False/moo_method=erm/num_tokens=10/multi_prompt=False/n_epochs=50/batch_size=16/lr=0.001/scheduler=False/weight_decay=0.0001/warmup=5/max_grad_norm=1/preference=None/seed=0/log_every=10/save_step=10/save_best=False/save_last=False/output_dir=outputs/data_dir=/vast/hvp2011/data/metashifts/MetaDatasetCatDog/test_data_dir=None/data_transform=AugWaterbirdsCelebATransform/dataset=MetaShiftDataset/alpha=0.0/model=imagenet_resnet50_pretrained/train_from_scratch=False/moo_method=erm/num_tokens=10/multi_prompt=False/n_epochs=50/batch_size=16/lr=0.001/scheduler=False/weight_decay=0.0001/warmup=5/max_grad_norm=1/preference=None/seed=0/log_every=10/save_step=10/save_best=False/save_last=False/stats.pt"
groupdro_ckpt = "outputs/data_dir=/vast/hvp2011/data/metashifts/MetaDatasetCatDog/test_data_dir=None/data_transform=AugWaterbirdsCelebATransform/dataset=MetaShiftDataset/alpha=1.0/model=imagenet_resnet50_pretrained/train_from_scratch=False/moo_method=groupdro/num_tokens=10/multi_prompt=False/n_epochs=50/batch_size=16/lr=0.001/scheduler=False/weight_decay=0.0001/warmup=5/max_grad_norm=1/preference=None/seed=0/log_every=10/save_step=10/save_best=False/save_last=False/output_dir=outputs/data_dir=/vast/hvp2011/data/metashifts/MetaDatasetCatDog/test_data_dir=None/data_transform=AugWaterbirdsCelebATransform/dataset=MetaShiftDataset/alpha=1.0/model=imagenet_resnet50_pretrained/train_from_scratch=False/moo_method=groupdro/num_tokens=10/multi_prompt=False/n_epochs=50/batch_size=16/lr=0.001/scheduler=False/weight_decay=0.0001/warmup=5/max_grad_norm=1/preference=None/seed=0/log_every=10/save_step=10/save_best=False/save_last=False/stats.pt"
our_ckpt = "outputs/data_dir=/vast/hvp2011/data/metashifts/MetaDatasetCatDog/test_data_dir=None/data_transform=AugWaterbirdsCelebATransform/dataset=MetaShiftDataset/alpha=1.2/model=imagenet_resnet50_pretrained/train_from_scratch=False/moo_method=epo/num_tokens=10/multi_prompt=False/n_epochs=50/batch_size=16/lr=0.003/scheduler=False/weight_decay=0.01/warmup=5/max_grad_norm=1/preference=['1', '1', '1', '1']/seed=0/log_every=10/save_step=10/save_best=False/save_last=False/output_dir=outputs/data_dir=/vast/hvp2011/data/metashifts/MetaDatasetCatDog/test_data_dir=None/data_transform=AugWaterbirdsCelebATransform/dataset=MetaShiftDataset/alpha=1.2/model=imagenet_resnet50_pretrained/train_from_scratch=False/moo_method=epo/num_tokens=10/multi_prompt=False/n_epochs=50/batch_size=16/lr=0.003/scheduler=False/weight_decay=0.01/warmup=5/max_grad_norm=1/preference=['1', '1', '1', '1']/seed=0/log_every=10/save_step=10/save_best=False/save_last=False/stats.pt"

# same config
# erm_ckpt = "outputs/data_dir=/vast/hvp2011/data/metashifts/MetaDatasetCatDog/test_data_dir=None/data_transform=AugWaterbirdsCelebATransform/dataset=MetaShiftDataset/alpha=0.0/model=imagenet_resnet50_pretrained/train_from_scratch=False/moo_method=erm/num_tokens=10/multi_prompt=False/n_epochs=50/batch_size=16/lr=0.01/scheduler=False/weight_decay=0.01/warmup=5/max_grad_norm=1/preference=None/seed=0/log_every=100/save_step=10/save_best=False/save_last=False/output_dir=outputs/data_dir=/vast/hvp2011/data/metashifts/MetaDatasetCatDog/test_data_dir=None/data_transform=AugWaterbirdsCelebATransform/dataset=MetaShiftDataset/alpha=0.0/model=imagenet_resnet50_pretrained/train_from_scratch=False/moo_method=erm/num_tokens=10/multi_prompt=False/n_epochs=50/batch_size=16/lr=0.01/scheduler=False/weight_decay=0.01/warmup=5/max_grad_norm=1/preference=None/seed=0/log_every=100/save_step=10/save_best=False/save_last=False/stats.pt"
# groupdro_ckpt = "outputs/data_dir=/vast/hvp2011/data/metashifts/MetaDatasetCatDog/test_data_dir=None/data_transform=AugWaterbirdsCelebATransform/dataset=MetaShiftDataset/alpha=1.0/model=imagenet_resnet50_pretrained/train_from_scratch=False/moo_method=groupdro/num_tokens=10/multi_prompt=False/n_epochs=50/batch_size=16/lr=0.01/scheduler=False/weight_decay=0.01/warmup=5/max_grad_norm=1/preference=None/seed=0/log_every=100/save_step=10/save_best=False/save_last=False/output_dir=outputs/data_dir=/vast/hvp2011/data/metashifts/MetaDatasetCatDog/test_data_dir=None/data_transform=AugWaterbirdsCelebATransform/dataset=MetaShiftDataset/alpha=1.0/model=imagenet_resnet50_pretrained/train_from_scratch=False/moo_method=groupdro/num_tokens=10/multi_prompt=False/n_epochs=50/batch_size=16/lr=0.01/scheduler=False/weight_decay=0.01/warmup=5/max_grad_norm=1/preference=None/seed=0/log_every=100/save_step=10/save_best=False/save_last=False/stats.pt"

erm_stats = torch.load(erm_ckpt)
groupdro_stats = torch.load(groupdro_ckpt)
our_stats = torch.load(our_ckpt)

for k, v in erm_stats.items():
    print(k)
    print(v.shape)

# group_losses
# torch.Size([50, 4])
# group_accs
# torch.Size([50, 4])
# group_conflicts
# torch.Size([50, 4])
N_EPOCHS = 100
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))
fig = plt.subplot(1, 3, 1)
plt.ylabel('Accuracy', fontsize=12)
plt.xlabel('Epoch', fontsize=12)
# #plt.ylim(0.75, 1)
plt.plot(erm_stats['group_accs'][:N_EPOCHS, 0], label='group 0')
plt.plot(erm_stats['group_accs'][:N_EPOCHS, 1], label='group 1')
plt.plot(erm_stats['group_accs'][:N_EPOCHS, 2], label='group 2')
plt.plot(erm_stats['group_accs'][:N_EPOCHS, 3], label='group 3')
plt.title('ERM', fontsize=12)

fig = plt.subplot(1, 3, 2)
plt.xlabel('Epoch', fontsize=12)
plt.title('GroupDRO', fontsize=12)
# #plt.ylim(0.75, 1)
plt.plot(groupdro_stats['group_accs'][:N_EPOCHS, 0], label='group 0')
plt.plot(groupdro_stats['group_accs'][:N_EPOCHS, 1], label='group 1')
plt.plot(groupdro_stats['group_accs'][:N_EPOCHS, 2], label='group 2')
plt.plot(groupdro_stats['group_accs'][:N_EPOCHS, 3], label='group 3')

# legend
plt.legend(loc='upper center')

fig = plt.subplot(1, 3, 3)
plt.xlabel('Epoch', fontsize=12)
# #plt.ylim(0.75, 1)
plt.plot(our_stats['group_accs'][:N_EPOCHS, 0], label='group 0')
plt.plot(our_stats['group_accs'][:N_EPOCHS, 1], label='group 1')
plt.plot(our_stats['group_accs'][:N_EPOCHS, 2], label='group 2')
plt.plot(our_stats['group_accs'][:N_EPOCHS, 3], label='group 3')
plt.title('Ours', fontsize=12)
plt.savefig('accs.pdf')
plt.show()

# conflict
# N_EPOCHS = 25

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))
fig = plt.subplot(1, 3, 1)
plt.ylabel('% Conflict', fontsize=12)
plt.xlabel('Epoch', fontsize=12)
plt.title('ERM', fontsize=12)
plt.plot(erm_stats['group_conflicts'][:N_EPOCHS, 0], label='group 0')
plt.plot(erm_stats['group_conflicts'][:N_EPOCHS, 1], label='group 1')
plt.plot(erm_stats['group_conflicts'][:N_EPOCHS, 2], label='group 2')
plt.plot(erm_stats['group_conflicts'][:N_EPOCHS, 3], label='group 3')
#plt.ylim(0, 1)

fig = plt.subplot(1, 3, 2)
plt.xlabel('Epoch', fontsize=12)
plt.title('GroupDRO', fontsize=12)
plt.plot(groupdro_stats['group_conflicts'][:N_EPOCHS, 0], label='group 0')
plt.plot(groupdro_stats['group_conflicts'][:N_EPOCHS, 1], label='group 1')
plt.plot(groupdro_stats['group_conflicts'][:N_EPOCHS, 2], label='group 2')
plt.plot(groupdro_stats['group_conflicts'][:N_EPOCHS, 3], label='group 3')
#plt.ylim(0, 1)

# legend
plt.legend(loc='upper center')

fig = plt.subplot(1, 3, 3)
plt.xlabel('Epoch', fontsize=12)
plt.title('Ours', fontsize=12)
plt.plot(our_stats['group_conflicts'][:N_EPOCHS, 0], label='group 0')
plt.plot(our_stats['group_conflicts'][:N_EPOCHS, 1], label='group 1')
plt.plot(our_stats['group_conflicts'][:N_EPOCHS, 2], label='group 2')
plt.plot(our_stats['group_conflicts'][:N_EPOCHS, 3], label='group 3')
# fig.get_legend().remove()
#plt.ylim(0, 1)

plt.savefig('conflict.pdf')
plt.show()


# N_EPOCHS = 25

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))
fig = plt.subplot(1, 3, 1)
plt.ylabel('Loss', fontsize=12)
plt.xlabel('Epoch', fontsize=12)
plt.title('ERM', fontsize=12)
plt.plot(erm_stats['group_losses'][:N_EPOCHS, 0], label='group 0')
plt.plot(erm_stats['group_losses'][:N_EPOCHS, 1], label='group 1')
plt.plot(erm_stats['group_losses'][:N_EPOCHS, 2], label='group 2')
plt.plot(erm_stats['group_losses'][:N_EPOCHS, 3], label='group 3')
#plt.ylim(0, .6)

fig = plt.subplot(1, 3, 2)
plt.xlabel('Epoch', fontsize=12)
plt.title('GroupDRO', fontsize=12)
plt.plot(groupdro_stats['group_losses'][:N_EPOCHS, 0], label='group 0')
plt.plot(groupdro_stats['group_losses'][:N_EPOCHS, 1], label='group 1')
plt.plot(groupdro_stats['group_losses'][:N_EPOCHS, 2], label='group 2')
plt.plot(groupdro_stats['group_losses'][:N_EPOCHS, 3], label='group 3')
#plt.ylim(0, .6)

# legend
plt.legend(loc='upper center', ncol=2)

fig = plt.subplot(1, 3, 3)
plt.xlabel('Epoch', fontsize=12)
plt.title('Ours', fontsize=12)
plt.plot(our_stats['group_losses'][:N_EPOCHS, 0], label='group 0')
plt.plot(our_stats['group_losses'][:N_EPOCHS, 1], label='group 1')
plt.plot(our_stats['group_losses'][:N_EPOCHS, 2], label='group 2')
plt.plot(our_stats['group_losses'][:N_EPOCHS, 3], label='group 3')
# fig.get_legend().remove()
#plt.ylim(0, .6)
plt.tight_layout()
plt.savefig('losses.pdf')
plt.show()