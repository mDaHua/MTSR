import numpy as np
import torch.utils.data as data
import scipy.io as sio
import torch
import os
import h5py
import utils


def is_mat_file(filename):
    return any(filename.endswith(extension) for extension in [".mat","h5"])

class HSTrainingData(data.Dataset):
    def __init__(self, image_dir, augment=None, use_3D=False):
        self.image_files = [os.path.join(image_dir, x) for x in os.listdir(image_dir) if is_mat_file(x)]
        self.augment = augment
        self.use_3Dconv = use_3D
        if self.augment:
            self.factor = 8
        else:
            self.factor = 1

    def __getitem__(self, index):
        file_index = index
        aug_num = 0
        if self.augment:
            file_index = index // self.factor   #
            aug_num = int(index % self.factor)  # 0-7
        load_dir = self.image_files[file_index]

        if load_dir.endswith('.mat'):
            data = sio.loadmat(load_dir)
            ms = np.array(data['ms'][...], dtype=np.float32)
            lms = np.array(data['ms_bicubic'][...], dtype=np.float32)
            gt = np.array(data['gt'][...], dtype=np.float32)
        elif load_dir.endswith('.h5'):
            with h5py.File(load_dir, 'r') as hf:
                ms = np.array(hf['ms'][...], dtype=np.float32)
                lms = np.array(hf['ms_bicubic'][...], dtype=np.float32)
                gt = np.array(hf['gt'][...], dtype=np.float32)

        ms, lms, gt = utils.data_augmentation(ms, mode=aug_num), utils.data_augmentation(lms, mode=aug_num), \
                        utils.data_augmentation(gt, mode=aug_num)

        if self.use_3Dconv:
            ms, lms, gt = ms[np.newaxis, :, :, :], lms[np.newaxis, :, :, :], gt[np.newaxis, :, :, :]
            ms = torch.from_numpy(ms.copy()).permute(0, 3, 1, 2)
            lms = torch.from_numpy(lms.copy()).permute(0, 3, 1, 2)
            gt = torch.from_numpy(gt.copy()).permute(0, 3, 1, 2)
        else:
            ms = torch.from_numpy(ms.copy()).permute(2, 0, 1)
            lms = torch.from_numpy(lms.copy()).permute(2, 0, 1)
            gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        return ms, lms, gt

    def __len__(self):
        return len(self.image_files)*self.factor

class HSTrainingData1(data.Dataset):
    def __init__(self, image_dir, augment=None, use_3D=False):
        self.augment = augment
        self.use_3Dconv = use_3D
        if self.augment:
            self.factor = 8
        else:
            self.factor = 1

        with h5py.File(image_dir, 'r') as hf:
            self.ms = np.array(hf['ms'][...], dtype=np.float32)
            self.lms = np.array(hf['ms_bicubic'][...], dtype=np.float32)
            self.gt = np.array(hf['gt'][...], dtype=np.float32)

    def __getitem__(self, index):
        file_index = index
        aug_num = 0
        if self.augment:
            file_index = index // self.factor   #
            aug_num = int(index % self.factor)  # 0-7

        gt = self.gt[index, :, :, :]
        ms = self.ms[index, :, :, :]
        lms = self.lms[index, :, :, :]

        ms, lms, gt = utils.data_augmentation(ms, mode=aug_num), utils.data_augmentation(lms, mode=aug_num), \
                        utils.data_augmentation(gt, mode=aug_num)

        if self.use_3Dconv:
            ms, lms, gt = ms[np.newaxis, :, :, :], lms[np.newaxis, :, :, :], gt[np.newaxis, :, :, :]
            ms = torch.from_numpy(ms.copy()).permute(0, 3, 1, 2)
            lms = torch.from_numpy(lms.copy()).permute(0, 3, 1, 2)
            gt = torch.from_numpy(gt.copy()).permute(0, 3, 1, 2)
        else:
            ms = torch.from_numpy(ms.copy()).permute(2, 0, 1)
            lms = torch.from_numpy(lms.copy()).permute(2, 0, 1)
            gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        return ms, lms, gt

    def __len__(self):
        return self.gt.shape[0]
