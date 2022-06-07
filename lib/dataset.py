import os
import numpy as np
from PIL import Image
import torch
import random
import scipy.io as sio
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import scipy.ndimage as snd

from lib.binvox_rw import read_as_3d_array
from lib.config import cfg
from lib.data_augmentation import preprocess_img

class DatasetLoader(Dataset):
    def __init__(self, train=True):

        if cfg.DATASET == 'shapenet':
            if train:
                self.samples = [os.path.join(cfg.DIR.SHNET_TRAIN % ('Rendering'), dname) for dname in os.listdir(cfg.DIR.SHNET_TRAIN % ('Rendering'))]
            else:
                self.samples = [os.path.join(cfg.DIR.SHNET_TEST % ('Rendering'), dname) for dname in os.listdir(cfg.DIR.SHNET_TEST % ('Rendering'))]
        else:
            if train:
                self.samples = [os.path.join(cfg.DIR.HSITE_TRAIN, dname) for dname in os.listdir(cfg.DIR.HSITE_TRAIN)]
            else:
                self.samples = [os.path.join(cfg.DIR.HSITE_TEST, dname) for dname in os.listdir(cfg.DIR.HSITE_TEST)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

class DatasetCollateFn(object):
    def __init__(self, train=True):
        self.train = train

    def __call__(self, batch):
        batch_size = len(batch)

        # set up constants
        img_h = cfg.CONST.IMG_H
        img_w = cfg.CONST.IMG_W
        n_vox = cfg.CONST.N_VOX

        # This is the maximum number of views
        n_views = cfg.CONST.N_VIEWS

        if cfg.DATASET == 'shapenet':
            random_id = random.sample(range(cfg.TRAIN.NUM_RENDERING), n_views)
        else:
            random_id = range(5)

        # This will be fed into the queue. create new batch everytime
        batch_img = np.zeros((n_views, batch_size, 3, img_h, img_w))
        batch_voxel = np.zeros((batch_size, 2, n_vox, n_vox, n_vox))

        zoom_rate = n_vox / 256

        # load each data instance
        for batch_id, sample in enumerate(batch):
            for view_id, image_id in enumerate(random_id):
                im = Image.open(os.path.join(sample, "{:03d}.png".format(image_id)))
                t_im = preprocess_img(im, self.train)

                # channel, height, width
                batch_img[view_id, batch_id, :, :, :] = t_im.transpose((2, 0, 1))

            if cfg.DATASET == 'shapenet':
                voxel_mat = getVoxelFromMat(os.path.join(sample.replace('/Rendering/', '/Voxels/'), "model.mat"))
                voxel_small = snd.zoom(voxel_mat, (zoom_rate, zoom_rate, zoom_rate))
                voxel_data = np.transpose(voxel_small, (0, 2, 1))
            else:
                basename = os.path.basename(sample)
                with open(os.path.join(sample, "%s.binvox" % basename), 'rb') as f:
                    voxel = read_as_3d_array(f)

                voxel_data = snd.zoom(voxel.data, (zoom_rate, zoom_rate, zoom_rate))

            batch_voxel[batch_id, 0, :, :, :] = voxel_data < 1
            batch_voxel[batch_id, 1, :, :, :] = voxel_data

        # float32 should be enough.
        batch_img = torch.from_numpy(batch_img.astype(np.float32))
        batch_voxel = torch.from_numpy(batch_voxel.astype(np.float32))

        return batch_img, batch_voxel


def getVoxelFromMat(path, cube_len=64):
    voxels = sio.loadmat(path)['input'][0]
    return voxels

def plotFromVoxels(voxels, title):
    z, x, y = voxels.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    ax.scatter(x, y, -z, zdir='z', c='red')
    plt.show()