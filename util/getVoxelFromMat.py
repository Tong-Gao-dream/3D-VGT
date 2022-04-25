import scipy.io as io
import numpy as np
import scipy.ndimage as nd

def getVoxelFromMat(path, cube_len=64):

    voxels = io.loadmat(path)['input']
    # voxels = io.loadmat(path)
    voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))
    if cube_len != 32 and cube_len == 64:
        voxels = nd.zoom(voxels, (2, 2, 2), mode='constant', order=0)
    return voxels
