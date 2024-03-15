import os
import torch
import numpy as np
from torch.utils.data import Dataset

import cv2 as cv
import h5py


class HDFDataset(Dataset):

    def __init__(self, hdf_file_path,
                 prefixes=None,
                 get_2nd_sample=False,
                 pick_exactly_per_person=None,
                 pick_at_least_per_person=None):
        assert os.path.isfile(hdf_file_path)
        self.get_2nd_sample = get_2nd_sample
        self.pick_exactly_per_person = pick_exactly_per_person
        self.hdf_path = hdf_file_path
        self.hdf = None  # h5py.File(hdf_file, 'r')

        with h5py.File(self.hdf_path, 'r', libver='latest', swmr=True) as h5f:
            hdf_keys = sorted(list(h5f.keys()))
            self.prefixes = hdf_keys if prefixes is None else prefixes
            if pick_exactly_per_person is not None:
                assert pick_at_least_per_person is None
                # Pick exactly x many entries from front of group
                self.prefixes = [
                    k for k in self.prefixes if k in h5f
                    and len(next(iter(h5f[k].values()))) >= pick_exactly_per_person
                ]
                self.index_to_query = sum([
                    [(prefix, i) for i in range(pick_exactly_per_person)]
                    for prefix in self.prefixes
                ], [])
            elif pick_at_least_per_person is not None:
                assert pick_exactly_per_person is None
                # Pick people for which there exists at least x many entries
                self.prefixes = [
                    k for k in self.prefixes if k in h5f
                    and len(next(iter(h5f[k].values()))) >= pick_at_least_per_person
                ]
                self.index_to_query = sum([
                    [(prefix, i) for i in range(len(next(iter(h5f[prefix].values()))))]
                    for prefix in self.prefixes
                ], [])
            else:
                # Pick all entries of person
                self.prefixes = [  # to address erroneous inputs
                    k for k in self.prefixes if k in h5f
                    and len(next(iter(h5f[k].values()))) > 0
                ]
                self.index_to_query = sum([
                    [(prefix, i) for i in range(len(next(iter(h5f[prefix].values()))))]
                    for prefix in self.prefixes
                ], [])

    def __len__(self):
        return len(self.index_to_query)

    def close_hdf(self):
        if self.hdf is not None:
            self.hdf.close()
            self.hdf = None
