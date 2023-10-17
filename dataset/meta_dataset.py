import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset



class MetaDatasetCatDog(Dataset):
    """
    MetaShift data. 
    `cat` is correlated with (`sofa`, `bed`), and `dog` is correlated with (`bench`, `bike`);
    In testing set, the backgrounds of both classes are `shelf`.        
    """
    def __init__(self,  basedir, split="train", transform=None):

        self.base_dir = basedir

        self.train_data_dir = os.path.join(self.base_dir, "train")
        self.test_data_dir = os.path.join(self.base_dir, 'test')

        self.transform = transform
        # Set training and testing environments
        self.n_classes = 2
        self.n_groups = 4
        cat_dict = {0: ["sofa"], 1: ["bed"]}
        dog_dict = {0: ['bench'], 1: ['bike']}
        self.test_groups = { "cat": ["shelf"], "dog": ["shelf"]}
        self.train_groups = {"cat": cat_dict, "dog": dog_dict}
        
        if split == "train":
            self.n_spurious = 4
            
            self.filename_array, self.group_array, self.y_array = self.get_data(self.train_groups,
                                                                                                is_training=True)
        else:
            self.test_filename_array, self.test_group_array, self.test_y_array = self.get_data(self.test_groups,
                                                                                            is_training=False)
            self.n_spurious = 1

            # split test and validation set
            np.random.seed(100)
            test_idxes = np.arange(len(self.test_group_array))
            val_idxes, _ = train_test_split(np.arange(len(test_idxes)), test_size=0.85, random_state=0)
            test_idxes = np.setdiff1d(test_idxes, val_idxes)
            
            all_idxes = val_idxes.tolist() + test_idxes.tolist()
            print(sorted(all_idxes) == np.arange(len(self.test_group_array)).tolist())
            
            if split == "val":
                self.filename_array, self.group_array, self.y_array = self.test_filename_array[val_idxes], self.test_group_array[val_idxes], self.test_y_array[val_idxes]
            elif split == "test": 
                self.filename_array, self.group_array, self.y_array = self.test_filename_array[test_idxes], self.test_group_array[test_idxes], self.test_y_array[test_idxes]
            else:
                raise ValueError(f"Invalid split: {split}")


        self.domains = self.group_array
        self.n_groups = len(np.unique(self.group_array))
        self._count_groups()
    def get_group_array(self):
        return self.group_array

    def get_label_array(self):
        return self.y_array


    def get_data(self, groups, is_training):
        filenames = []
        group_ids = []
        ys = []
        id_count = 0
        animal_count = 0
        for animal in groups.keys():
            if is_training:
                for _, group_animal_data in groups[animal].items():
                    for group in group_animal_data:
                        for file in os.listdir(f"{self.train_data_dir}/{animal}/{animal}({group})"):
                            filenames.append(os.path.join(f"{self.train_data_dir}/{animal}/{animal}({group})", file))
                            group_ids.append(id_count)
                            ys.append(animal_count)
                    id_count += 1
            else:
                for group in groups[animal]:
                    for file in os.listdir(f"{self.test_data_dir}/{animal}/{animal}({group})"):
                        filenames.append(os.path.join(f"{self.test_data_dir}/{animal}/{animal}({group})", file))
                        group_ids.append(id_count)
                        ys.append(animal_count)
                    id_count += 1
            animal_count += 1
        return np.array(filenames), np.array(group_ids), np.array(ys)


    def _count_groups(self):
        self.group_counts = self._bincount_array_as_tensor(self.group_array)
        self.n_groups = len(self.group_counts)
        self.group_ratio = self.group_counts / self.group_counts.sum()

    @staticmethod
    def _bincount_array_as_tensor(arr):
        return torch.from_numpy(np.bincount(arr)).long()

    def __len__(self):
        return len(self.group_array)

    def __getitem__(self, idx):
        g = self.group_array[idx]
        y = self.y_array[idx]
        x = self.get_image(idx)
        # return x, y, g, idx
        return x, y, g, None
        


    def get_image(self, idx):
        img_filename = self.filename_array[idx]
        img = Image.open(img_filename).convert("RGB")
           
        if self.transform:
            img = self.transform(img)
        return img

if __name__ == "__main__":
    from data_transforms import AugWaterbirdsCelebATransform
    transform = AugWaterbirdsCelebATransform(train=False)
    val_ds = MetaDatasetCatDog("/scratch/hvp2011/implement/spurious-correlation/data/metashifts/MetaDatasetCatDog", split="val", transform=transform)
    print(val_ds[0])
    test_ds = MetaDatasetCatDog("/scratch/hvp2011/implement/spurious-correlation/data/metashifts/MetaDatasetCatDog", split="test", transform=transform)
    print(test_ds[0])    
    
    