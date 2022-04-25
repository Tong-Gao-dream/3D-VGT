import os
from torch.utils.data import DataLoader
import cv2
import numpy as np
import scipy.io as io
import scipy.ndimage as nd
import torch
from torch.utils.data import Dataset
from getVoxelFromMat import getVoxelFromMat

class ShapeNet_img(Dataset):
    def __init__(self, img_path):
        super(ShapeNet_img, self).__init__()
        self.images = []
        self.voxel = []
        files_img = os.listdir(img_path)

        for i in range(len(files_img)):
            img_file1 = os.path.join(img_path, files_img[i])

            files1_img = os.listdir(img_file1)

            for j in range(len(files1_img)):
                img_file2 = os.path.join(img_file1, files1_img[j])
                self.images.append(img_file2)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # img
        image_path = self.images[idx]
        img = cv2.imread(image_path)

        img = np.float32(img) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img)


        return img


class ShapeNet_voxel(Dataset):
    def __init__(self, voxel_path):
        super(ShapeNet_voxel, self).__init__()
        self.voxel = []

        files_voxel = os.listdir(voxel_path)

        for i in range(len(files_voxel)):
            voxel_file1 = os.path.join(voxel_path, files_voxel[i])

            files2_voxel = os.listdir(voxel_file1)


            voxel_file2 = os.path.join(voxel_file1, files2_voxel[0])
            self.voxel.append(voxel_file2)

    def __len__(self):
        return len(self.voxel)

    def __getitem__(self, idx):
        # voxel
        vox_path = self.voxel[idx]
        volume = np.asarray(getVoxelFromMat(vox_path, cube_len=32), dtype=np.float32)
        volume = volume[1:2, :, :, :]

        volume = volume[:, 0:258:2, 0:258:2, 0:258:2]

        vox = torch.FloatTensor(volume)

        return vox
