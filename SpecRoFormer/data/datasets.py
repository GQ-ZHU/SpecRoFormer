import numpy as np
import numpy.typing as npt
from torch.utils.data import Dataset
import random
from SpecRoFormer.data.const import SpecialToken
from SpecRoFormer.type import TokenSequence, AugmentationConfig, DefaultAugmentationConifg

class TrainDataset(Dataset):
    def __init__(
        self,
        data: dict[str, list[TokenSequence]],
        keys: npt.NDArray,
        n_views: int,
        is_augment: bool = False,
        augment_config: AugmentationConfig = DefaultAugmentationConifg
    ) -> None:
        """
            Training dataset
            Parameters:
            ---
            -   data: divided based on smiles (includes training set/validation set)
            -   keys: smiles used for the training set
            -   n_views: number of data augmentations or samples with the same smiles
            -   is_augment: whether to use data augmentation
            -   augment_config: data augmentation configuration file
        """
        super(TrainDataset, self).__init__()
        self._data = data
        self._keys = keys
        self._length = len(keys)
        self._n_views = n_views
        self.is_augment = is_augment
        self.augment_config = augment_config

    def augmentation(self):
        pass

    @property
    def length(self):
        """
            Dataset length
            
            The length can be manually set as a multiple of the lengths of the smiles used in the training set.
        """
        return self._length * 10

    def aug(self, item: TokenSequence):
        """
            Data augmentation 
            
            precursor.

            1. Randomly mask low-intensity values (intensity < config["removal_intensity"])
                The masking ratio is config["removal_max"].
            2. Randomly shift intensities 
                each peak is shifted by a different proportion, 
                with each peak being randomly shifted ± config["rate_intensity"] * raw_intensity.
        """
        seq_len = len(item["mz"])
        mz, intensity, mask = item["mz"], item["intensity"], item["mask"]

        # print(mz, intensity, mask)

        precursor_mz, precursor_intensity, precursor_mask = [
            mz[0]], [intensity[0]], [mask[0]]

        mz, intensity, mask = mz[1:], intensity[1:], mask[1:]

        # only for peaks
        indices = np.arange(len(mz))

        # 1. randomly remove the peaks with low intensity
        candidate_indices = np.where(
            (intensity < self.augment_config["removal_intensity"]) & (~mask)
        )[0]

        removal_percent = self.augment_config["removal_max"]
        # print(removal_percent)

        removed_indices = np.random.choice(
            candidate_indices,
            int(
                np.floor(
                    removal_percent * len(candidate_indices)
                )
            ),
            replace=False
        )
        # print(removed_indices)

        if len(removed_indices) > 0:
            indices = np.delete(indices, removed_indices)

        # print(indices)

        mz, intensity, mask = mz[indices], intensity[indices], mask[indices]
        # print(mz, intensity, mask)

        # 2. randomly change the peak intensity
        intensity = (
            1 - self.augment_config["rate_intensity"] *
            2 * (np.random.random(intensity.shape)-0.5)
        ) * intensity

        intensity = intensity / intensity.max()
        mz = np.concatenate((precursor_mz, mz))
        # print(mz.shape[0])
        intensity = np.concatenate((precursor_intensity, intensity))
        mask = np.concatenate((precursor_mask, mask))

        if len(mz) < seq_len:
            mz = np.pad(
                mz, (0, seq_len - len(mz)),
                mode='constant', constant_values=SpecialToken["PAD"]
            )

            intensity = np.pad(
                intensity, (0, seq_len - len(intensity)),
                mode='constant', constant_values=SpecialToken["PAD"]
            )

            mask = np.pad(
                mask, (0, seq_len - len(mask)),
                mode='constant', constant_values=True
            )

        return TokenSequence(
            mz=np.array(mz, dtype=np.float32),
            intensity=np.array(intensity, dtype=np.float32),
            mask=mask,
            smiles=item["smiles"]
        )

    def aug_plus(self, item: TokenSequence):
        """
        Data augmentation for mass spectra with explicit sub-spectrum consistency.

        Augmentation strategy:
        1. Explicit sub-spectrum construction (random peak dropping, structure-level)
        2. Low-intensity peak removal (noise-aware, secondary)
        3. Peak-wise intensity jitter
        """

        seq_len = len(item["mz"])
        mz = item["mz"].copy()
        intensity = item["intensity"].copy()
        mask = item["mask"].copy()

        # --------------------------------------------------
        # Separate precursor token (index 0)
        # --------------------------------------------------
        precursor_mz = mz[0:1]
        precursor_intensity = intensity[0:1]
        precursor_mask = mask[0:1]

        mz = mz[1:]
        intensity = intensity[1:]
        mask = mask[1:]

        # ==================================================
        # (A) Sub-spectrum augmentation (core consistency)
        # ==================================================
        if np.random.rand() < self.augment_config.get("subspec_prob", 0.0):
            valid_idx = np.where(~mask)[0]

            if len(valid_idx) > 0:
                keep_num = max(
                    1,
                    int(len(valid_idx) * (1 - self.augment_config["subspec_drop_ratio"]))
                )
                keep_idx = np.random.choice(valid_idx, keep_num, replace=False)

                new_mask = np.ones_like(mask)
                new_mask[keep_idx] = False

                intensity = intensity.copy()
                intensity[new_mask] = 0.0
                mask = new_mask

        # ==================================================
        # (B) Low-intensity peak removal (noise-aware, optional)
        # ==================================================
        if np.random.rand() < self.augment_config.get("low_intensity_prob", 0.0):
            candidate_indices = np.where(
                (intensity < self.augment_config["removal_intensity"]) & (~mask)
            )[0]

            if len(candidate_indices) > 0:
                removal_percent = self.augment_config["removal_max"]
                remove_num = int(np.floor(removal_percent * len(candidate_indices)))

                if remove_num > 0:
                    removed_indices = np.random.choice(
                        candidate_indices,
                        remove_num,
                        replace=False
                    )
                    mask[removed_indices] = True
                    intensity[removed_indices] = 0.0

        # ==================================================
        # (C) Peak-wise intensity jitter
        # ==================================================
        if self.augment_config["rate_intensity"] > 0:
            jitter = (
                1
                - self.augment_config["rate_intensity"]
                * 2
                * (np.random.random(intensity.shape) - 0.5)
            )
            intensity = intensity * jitter

            max_int = intensity.max()
            if max_int > 0:
                intensity = intensity / max_int

        # --------------------------------------------------
        # Re-attach precursor token
        # --------------------------------------------------
        mz = np.concatenate((precursor_mz, mz))
        intensity = np.concatenate((precursor_intensity, intensity))
        mask = np.concatenate((precursor_mask, mask))

        # --------------------------------------------------
        # Padding to original sequence length
        # --------------------------------------------------
        if len(mz) < seq_len:
            pad_len = seq_len - len(mz)

            mz = np.pad(
                mz,
                (0, pad_len),
                mode="constant",
                constant_values=SpecialToken["PAD"]
            )

            intensity = np.pad(
                intensity,
                (0, pad_len),
                mode="constant",
                constant_values=SpecialToken["PAD"]
            )

            mask = np.pad(
                mask,
                (0, pad_len),
                mode="constant",
                constant_values=True
            )

        return TokenSequence(
            mz=np.array(mz, dtype=np.float32),
            intensity=np.array(intensity, dtype=np.float32),
            mask=mask,
            smiles=item["smiles"]
        )

    def __len__(self):
        return self.length


    def __getitem__(self, index):
        """
        Perform a modulo operation on the index with respect to the training keys
            since we default to using 10 * the number of smiles, 
            the index may exceed len(keys)
        
        If data augmentation is applied and the specified probability condition is met, 
        randomly select one data entry from the dataset corresponding to the key, 
        apply two different data augmentation operations to it, and return the results.

        If data augmentation is not applied or the probability condition is not met, 
        randomly select two data entries from the set and treat them as separately augmented versions of the data.
        """
        label = self._keys[index % self._length]
        item_seq = self._data[label]
        data = []

        only_one = (len(item_seq) == 1)

        # if self.is_augment and np.random.random() < self.augment_config["prob"]:
        if self.is_augment and (only_one or np.random.random() < self.augment_config["prob"]):
            if only_one:
                item = item_seq[0]
            else:
                item = item_seq[int(np.random.choice(len(item_seq), 1))]
            for _ in range(self._n_views):
                aug = self.aug_plus(item)
                data.append([
                    aug["mz"],
                    aug["intensity"],
                    aug["mask"]
                ])

        else:
            for _ in range(self._n_views):
                item = item_seq[int(np.random.choice(
                    len(self._data[label]), 1))]
                data.append([
                    item['mz'],
                    item['intensity'],
                    item['mask']
                ])
        return data, label
    

class TestDataset(Dataset):
    def __init__(self, sequences: list[TokenSequence]) -> None:
        super(TestDataset, self).__init__()
        self._sequences = sequences
        self.length = len(sequences)

    def __len__(self):
        return self.length

    def __getitem__(self, index: int):
        sequence = self._sequences[index]
        return sequence["mz"], sequence["intensity"], sequence["mask"]