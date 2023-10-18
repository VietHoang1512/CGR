import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10

try:
    import wilds
    from wilds.datasets.wilds_dataset import WILDSSubset

    has_wilds = True
except:
    has_wilds = False


def _get_split(split):
    try:
        return ["train", "val", "test"].index(split)
    except ValueError:
        raise (f"Unknown split {split}")


def _cast_int(arr):
    if isinstance(arr, np.ndarray):
        return arr.astype(int)
    elif isinstance(arr, torch.Tensor):
        return arr.int()
    else:
        raise NotImplementedError


class SpuriousCorrelationDataset(Dataset):
    def __init__(self, basedir, split="train", transform=None):
        self.basedir = basedir
        self.metadata_df = self._get_metadata(split)

        self.transform = transform

        # TODO: Check this carefully before running/committing
        self.y_array = self.metadata_df["y"].values
        # self.y_array = self.metadata_df["place"].values

        self.spurious_array = self.metadata_df["place"].values

        self._count_attributes()
        if "group" in self.metadata_df:
            self.group_array = self.metadata_df["group"].values
        else:
            self._get_class_spurious_groups()
        self._count_groups()
        self.text = not "img_filename" in self.metadata_df
        if self.text:
            print("NLP dataset")
            self.text_array = list(
                pd.read_csv(os.path.join(basedir, "text.csv"))["text"]
            )
        else:
            self.filename_array = self.metadata_df["img_filename"].values

    def _get_metadata(self, split):
        split_i = _get_split(split)
        try:
            metadata_df = pd.read_csv(os.path.join(self.basedir, "celeba_metadata.csv"))
        except FileNotFoundError as e:
            print("Exception:", e)
            print("celeba_metadata.csv not found, using metadata.csv")
            metadata_df = pd.read_csv(os.path.join(self.basedir, "metadata.csv"))
            
        metadata_df = metadata_df[metadata_df["split"] == split_i]
        return metadata_df

    def _count_attributes(self):
        self.n_classes = np.unique(self.y_array).size
        self.n_spurious = np.unique(self.spurious_array).size
        self.y_counts = self._bincount_array_as_tensor(self.y_array)
        self.spurious_counts = self._bincount_array_as_tensor(self.spurious_array)

    def _count_groups(self):
        self.group_counts = self._bincount_array_as_tensor(self.group_array)
        # self.n_groups = np.unique(self.group_array).size
        self.n_groups = len(self.group_counts)
        self.group_ratio = self.group_counts / self.group_counts.sum()

    def _get_class_spurious_groups(self):
        self.group_array = _cast_int(
            self.y_array * self.n_spurious + self.spurious_array
        )

    @staticmethod
    def _bincount_array_as_tensor(arr):
        return torch.from_numpy(np.bincount(arr)).long()

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.group_array[idx]
        s = self.spurious_array[idx]
        if self.text:
            x = self._text_getitem(idx)
        else:
            x = self._image_getitem(idx)
        return x, y, g, s

    def _image_getitem(self, idx):
        img_path = os.path.join(self.basedir, self.filename_array[idx])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

    def _text_getitem(self, idx):
        text = self.text_array[idx]
        if self.transform:
            text = self.transform(text)
        return text


class MultiNLIDataset(SpuriousCorrelationDataset):
    """Adapted from https://github.com/kohpangwei/group_DRO/blob/master/data/multinli_dataset.py"""

    def __init__(self, basedir, split="train", transform=None):
        assert transform is None, "transfrom should be None"

        # utils_glue module in basedir is needed to load data
        import sys

        sys.path.append(basedir)

        self.basedir = basedir
        self.metadata_df = pd.read_csv(
            os.path.join(self.basedir, "metadata_random.csv")
        )
        bert_filenames = [
            "cached_train_bert-base-uncased_128_mnli",
            "cached_dev_bert-base-uncased_128_mnli",
            "cached_dev_bert-base-uncased_128_mnli-mm",
        ]
        features_array = sum(
            [torch.load(os.path.join(self.basedir, name)) for name in bert_filenames],
            start=[],
        )
        all_input_ids = torch.tensor([f.input_ids for f in features_array]).long()
        all_input_masks = torch.tensor([f.input_mask for f in features_array]).long()
        all_segment_ids = torch.tensor([f.segment_ids for f in features_array]).long()
        # all_label_ids = torch.tensor([
        #     f.label_id for f in self.features_array]).long()

        split_i = _get_split(split)
        split_mask = (self.metadata_df["split"] == split_i).values

        self.x_array = torch.stack(
            (all_input_ids, all_input_masks, all_segment_ids), dim=2
        )[split_mask]
        self.metadata_df = self.metadata_df[split_mask]
        self.y_array = self.metadata_df["gold_label"].values
        self.spurious_array = self.metadata_df["sentence2_has_negation"].values
        self._count_attributes()
        self._get_class_spurious_groups()
        self._count_groups()

    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.group_array[idx]
        s = self.spurious_array[idx]
        x = self.x_array[idx]
        return x, y, g, s


class DeBERTaMultiNLIDataset(MultiNLIDataset):
    def __init__(self, basedir, split="train", transform=None):
        assert transform is None, "transfrom should be None"

        self.basedir = basedir
        self.metadata_df = pd.read_csv(
            os.path.join(self.basedir, "metadata_random.csv")
        )
        self.basedir = basedir
        split_i = _get_split(split)
        split_mask = (self.metadata_df["split"] == split_i).values
        self.x_array = torch.load(
            os.path.join(self.basedir, "cached_deberta-base_220_mnli")
        )[split_mask]
        self.metadata_df = self.metadata_df[split_mask]
        self.y_array = self.metadata_df["gold_label"].values
        self.spurious_array = self.metadata_df["sentence2_has_negation"].values
        self._count_attributes()
        self._get_class_spurious_groups()
        self._count_groups()


class BERTMultilingualMultiNLIDataset(MultiNLIDataset):
    def __init__(self, basedir, split="train", transform=None):
        assert transform is None, "transfrom should be None"

        self.basedir = basedir
        self.metadata_df = pd.read_csv(
            os.path.join(self.basedir, "metadata_random.csv")
        )
        self.basedir = basedir
        split_i = _get_split(split)
        split_mask = (self.metadata_df["split"] == split_i).values
        self.x_array = torch.load(
            os.path.join(self.basedir, "cached_bert-base-multilingual_150_mnli")
        )[split_mask]
        self.metadata_df = self.metadata_df[split_mask]
        self.y_array = self.metadata_df["gold_label"].values
        self.spurious_array = self.metadata_df["sentence2_has_negation"].values
        self._count_attributes()
        self._get_class_spurious_groups()
        self._count_groups()


class BaseWildsDataset(SpuriousCorrelationDataset):
    def __init__(self, ds_name, basedir, split, transform, y_name, spurious_name):
        assert has_wilds, "wilds package not found"
        self.basedir = basedir
        self.root_dir = "/".join(self.basedir.split("/")[:-2])
        base_dataset = wilds.get_dataset(
            dataset=ds_name, download=False, root_dir=self.root_dir
        )
        self.dataset = base_dataset.get_subset(split, transform=transform)

        column_names = self.dataset.metadata_fields
        if y_name:
            y_idx = column_names.index(y_name)
            self.y_array = self.dataset.metadata_array[:, y_idx]
        if spurious_name:
            s_idx = column_names.index(spurious_name)
            self.spurious_idx = s_idx
            self.spurious_array = self.dataset.metadata_array[:, s_idx]
        if y_name and spurious_name:
            self._count_attributes()

    def __getitem__(self, idx):
        x, y, metadata = self.dataset[idx]
        s = metadata[self.spurious_idx]
        return x, y, s, s

    def __len__(self):
        return len(self.dataset)


class WildsFMOW(BaseWildsDataset):
    def __init__(self, basedir, split="train", transform=None):
        super().__init__("fmow", basedir, split, transform, "y", "region")
        self.group_array = self.spurious_array
        self._count_groups()


class WildsPoverty(BaseWildsDataset):
    # TODO(izmailovpavel): test and implement regression training
    def __init__(self, basedir, split="train", transform=None):
        # assert transform is None, "transfrom should be None"
        super().__init__("poverty", basedir, split, transform, "y", "urban")
        self.n_classes = None
        self.group_array = self.spurious_array
        self._count_groups()


class WildsCivilCommentsCoarse(BaseWildsDataset):
    def __init__(self, basedir, split="train", transform=None):
        super().__init__("civilcomments", basedir, split, transform, "y", None)
        attributes = [
            "male",
            "female",
            "LGBTQ",
            "black",
            "white",
            "christian",
            "muslim",
            "other_religions",
        ]
        column_names = self.dataset.metadata_fields
        self.spurious_cols = [column_names.index(a) for a in attributes]
        self.spurious_array = self.get_spurious(self.dataset.metadata_array)
        self._count_attributes()
        self._get_class_spurious_groups()
        self._count_groups()

    def get_spurious(self, metadata):
        if len(metadata.shape) == 1:
            return metadata[self.spurious_cols].sum(-1).clip(max=1)
        else:
            return metadata[:, self.spurious_cols].sum(-1).clip(max=1)

    def __getitem__(self, idx):
        x, y, metadata = self.dataset[idx]
        s = self.get_spurious(metadata)
        g = y * self.n_spurious + s
        return x, y, g, s


class WildsCivilCommentsCoarseNM(WildsCivilCommentsCoarse):
    def __init__(self, basedir, split="train", transform=None):
        super().__init__(basedir, split, transform)
        if split == "train":
            identities_mentioned = self.spurious_array > 0
            toxic = self.y_array == 1
            mask = (identities_mentioned & toxic) | (~identities_mentioned & ~toxic)
            train_idx = self.dataset.indices.copy()[mask]
            self.dataset = WILDSSubset(
                self.dataset.dataset,
                indices=train_idx,
                transform=self.dataset.transform,
            )
            self.spurious_array = self.get_spurious(self.dataset.metadata_array)
            self.y_array = self.y_array[mask]
            self._count_attributes()
            self._get_class_spurious_groups()
            self._count_groups()


class FakeSpuriousCIFAR10(SpuriousCorrelationDataset):
    """CIFAR10 with SpuriousCorrelationDataset API.

    Groups are the same as classes.
    """

    def __init__(self, basedir, split, transform=None, val_size=5000):
        split_i = _get_split(split)
        self.ds = CIFAR10(
            root=basedir, train=(split_i != 2), download=True, transform=transform
        )
        if split_i == 0:
            self.ds.data = self.ds.data[:-val_size]
            self.ds.targets = self.ds.targets[:-val_size]
        elif split_i == 1:
            self.ds.data = self.ds.data[-val_size:]
            self.ds.targets = self.ds.targets[-val_size:]

        self.y_array = np.array(self.ds.targets)
        self.n_classes = 10
        self.spurious_array = np.zeros_like(self.y_array)
        self.n_spurious = 1
        self.group_array = self.y_array

        self.n_groups = 10
        self.group_counts = self._bincount_array_as_tensor(self.group_array)
        self.y_counts = self._bincount_array_as_tensor(self.y_array)
        self.spurious_counts = self._bincount_array_as_tensor(self.spurious_array)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, y = self.ds[idx]
        return x, y, y, 0


def remove_minority_groups(trainset, num_remove):
    if num_remove == 0:
        return
    print("Removing minority groups")
    print("Initial groups", np.bincount(trainset.group_array))
    num_groups = np.bincount(trainset.group_array).size
    group_counts = trainset.group_counts
    minority_groups = np.argsort(group_counts.numpy())[:num_remove]
    idx = np.where(
        np.logical_and.reduce(
            [trainset.group_array != g for g in minority_groups], initial=True
        )
    )[0]
    trainset.x_array = trainset.x_array[idx]
    trainset.y_array = trainset.y_array[idx]
    trainset.group_array = trainset.group_array[idx]
    trainset.spurious_array = trainset.spurious_array[idx]
    if hasattr(trainset, "filename_array"):
        trainset.filename_array = trainset.filename_array[idx]
    trainset.metadata_df = trainset.metadata_df.iloc[idx]
    trainset.group_counts = torch.from_numpy(
        np.bincount(trainset.group_array, minlength=num_groups)
    )
    print("Final groups", np.bincount(trainset.group_array))


def balance_groups(ds):
    print("Original groups", ds.group_counts)
    group_counts = ds.group_counts.long().numpy()
    min_group = np.min(group_counts)
    group_idx = [np.where(ds.group_array == g)[0] for g in range(ds.n_groups)]
    for idx in group_idx:
        np.random.shuffle(idx)
    group_idx = [idx[:min_group] for idx in group_idx]
    idx = np.concatenate(group_idx, axis=0)
    ds.y_array = ds.y_array[idx]
    ds.group_array = ds.group_array[idx]
    ds.spurious_array = ds.spurious_array[idx]
    ds.filename_array = ds.filename_array[idx]
    ds.metadata_df = ds.metadata_df.iloc[idx]
    ds.group_counts = torch.from_numpy(np.bincount(ds.group_array))
    print("Final groups", ds.group_counts)


class MetaShiftDataset(Dataset):
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
        self.n_spurious = self.n_groups
        self.spurious_array = self.group_array
        # self.n_groups = len(np.unique(self.group_array))
    @staticmethod
    def _bincount_array_as_tensor(arr):
        return torch.from_numpy(np.bincount(arr)).long()

    def __len__(self):
        return len(self.group_array)

    def __getitem__(self, idx):
        g = self.group_array[idx]
        y = self.y_array[idx]
        x = self.get_image(idx)
        return x, y, g, g
        # return x, y, g, None
        


    def get_image(self, idx):
        img_filename = self.filename_array[idx]
        img = Image.open(img_filename).convert("RGB")
           
        if self.transform:
            img = self.transform(img)
        return img

class ISICDataset(Dataset):
    """
    ISIC dataset      
    """
    def __init__(self, basedir, split="train", transform=None, id_val=True):

        self.split_dir = os.path.join(basedir, 'trap-sets')
        self.data_dir = os.path.join(basedir, 'ISIC2018_Task1-2_Training_Input')
        self.transform = transform
        
        GROUP = 5 # following https://github.com/Wuyxin/DISC/blob/master/scripts/isic.sh
        metadata = {}
        metadata['train'] = pd.read_csv(os.path.join(self.split_dir, f'isic_annotated_train{GROUP}.csv'))
        if id_val:
            test_val_data = pd.read_csv(os.path.join(self.split_dir, f'isic_annotated_test{GROUP}.csv'))
            idx_val, idx_test = train_test_split(np.arange(len(test_val_data)), 
                                                test_size=0.8, random_state=0)
            metadata['test'] = test_val_data.iloc[idx_test]
            metadata['val'] = test_val_data.iloc[idx_val]
            
        else:
            metadata['test'] = pd.read_csv(os.path.join(self.split_dir, f'isic_annotated_test{GROUP}.csv'))
            metadata['val'] = pd.read_csv(os.path.join(self.split_dir, f'isic_annotated_val{GROUP}.csv'))
            # subtracting two dataframes 
            metadata_new = metadata['train'].merge(metadata['val'], how='left', indicator=True)
            metadata_new = metadata_new[metadata_new['_merge'] == 'left_only']
            metadata['train'] = metadata_new.drop(columns=['_merge'])
        
        confounder = 'hair'
        
        self.filename_array = np.array(metadata[split]['image'] )
        self.spurious_array = np.array(metadata[split][confounder])
        self.y_array = np.array(metadata[split]["label"])      
        self._count_attributes()
        self._get_class_spurious_groups()
        self._count_groups()        
        
    def _count_attributes(self):
        self.n_classes = np.unique(self.y_array).size
        self.n_spurious = np.unique(self.spurious_array).size
        self.y_counts = self._bincount_array_as_tensor(self.y_array)
        self.spurious_counts = self._bincount_array_as_tensor(self.spurious_array)

    def _count_groups(self):
        self.group_counts = self._bincount_array_as_tensor(self.group_array)
        # self.n_groups = np.unique(self.group_array).size
        self.n_groups = len(self.group_counts)
        self.group_ratio = self.group_counts / self.group_counts.sum()

    def _get_class_spurious_groups(self):
        self.group_array = _cast_int(
            self.y_array * self.n_spurious + self.spurious_array
        )

    @staticmethod
    def _bincount_array_as_tensor(arr):
        return torch.from_numpy(np.bincount(arr)).long()

    def __len__(self):
        return len(self.y_array)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.group_array[idx]
        s = self.spurious_array[idx]
        x = self._image_getitem(idx)
        return x, y, g, s

    def _image_getitem(self, idx):
        img_path = os.path.join(self.data_dir, self.filename_array[idx])[:-4] + '.jpg'
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img
    
if __name__ == "__main__":
    from data_transforms import AugWaterbirdsCelebATransform
    transform = AugWaterbirdsCelebATransform(train=True)
    val_ds = MetaShiftDataset("/scratch/hvp2011/implement/spurious-correlation/data/metashifts/MetaDatasetCatDog", split="train", transform=transform)
    for datum in val_ds:
        print(datum)  
