import glob
import logging
import numpy as np
import os
import random
import torch
import torch.utils.data
import h5py
from tqdm import tqdm

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class ImNetImageSamples(torch.utils.data.Dataset):
    def __init__(self,
                 data_path: str,
                 auto_encoder=None,
                 max_batch=32,
                 sample_interval=1,
                 image_idx=None,
                 sample_voxel_size: int = 64,
                 ):
        super(ImNetImageSamples, self).__init__()
        data_dict = h5py.File(data_path, 'r')
        self.data_voxels = data_dict['voxels'][:]
        self.data_voxels = np.reshape(self.data_voxels, [-1, 1, self.data_voxels.shape[1], self.data_voxels.shape[2],
                                                         self.data_voxels.shape[3]])
        self.data_values = data_dict['values_' + str(sample_voxel_size)][:].astype(np.float32)
        self.data_points = (data_dict['points_' + str(sample_voxel_size)][:].astype(np.float32) + 0.5) / 256 - 0.5

        ### get file
        label_txt_path = data_path[:-5] + '.txt'
        self.obj_paths = [os.path.basename(line).rstrip('\n') for line in open(label_txt_path, mode='r').readlines()]

        ### extract the latent vector
        if auto_encoder is not None:
            self.extract_latent_vector(data_voxels = self.data_voxels, auto_encoder = auto_encoder, max_batch = max_batch)

        ### interval
        self.sample_interval = sample_interval

        ### views num
        self.view_num = 24
        self.view_size = 137
        self.crop_size = 128
        self.crop_edge = self.view_size - self.crop_size

        ### pixels
        self.crop_size = 128
        offset_x = int(self.crop_edge / 2)
        offset_y = int(self.crop_edge / 2)
        self.data_pixels = np.reshape(
            data_dict['pixels'][:, :, offset_y:offset_y + self.crop_size, offset_x:offset_x + self.crop_size],
            [-1, self.view_num, 1, self.crop_size, self.crop_size])

        self.image_idx = image_idx

    def __len__(self):
        return self.data_pixels.shape[0] // self.sample_interval

    def __getitem__(self, idx):

        idx = idx * self.sample_interval

        if self.image_idx is None:
            view_index = np.random.randint(0, self.view_num)
        else:
            view_index = self.image_idx

        image = self.data_pixels[idx, view_index].astype(np.float32) / 255.0


        if hasattr(self, 'latent_vectors'):
            latent_vector_gt = self.latent_vectors[idx]
        else:
            latent_vector_gt = None

        processed_inputs = image, latent_vector_gt

        return processed_inputs, idx


    def extract_latent_vector(self, data_voxels,  auto_encoder, max_batch):


        num_batch = int(np.ceil(data_voxels.shape[0] / max_batch))

        results = []
        print("start to extract GT!!!")
        with tqdm(range(num_batch), unit='batch') as tlist:
            for i in tlist:
                batched_voxels = data_voxels[i*max_batch:(i+1)*max_batch].astype(np.float32)
                batched_voxels = torch.from_numpy(batched_voxels).float().to(device)

                latent_vectors = auto_encoder.encoder(batched_voxels).detach().cpu().numpy()
                results.append(latent_vectors)

        if len(results) == 1:
            self.latent_vectors = results
        else:
            self.latent_vectors = np.concatenate(tuple(results), axis = 0)
        print("Done the extraction of GT!!!")



class ImNetSamples(torch.utils.data.Dataset):
    def __init__(self,
                 data_path: str,
                 sample_voxel_size: int,
                 interval=1):
        super(ImNetSamples, self).__init__()
        self.sample_voxel_size = sample_voxel_size
        data_dict = h5py.File(data_path, 'r')
        self.data_points = (data_dict['points_' + str(self.sample_voxel_size)][:].astype(np.float32) + 0.5) / 256 - 0.5
        self.data_values = data_dict['values_' + str(self.sample_voxel_size)][:].astype(np.float32)
        self.data_voxels = data_dict['voxels'][:]
        self.data_voxels = np.reshape(self.data_voxels, [-1, 1, self.data_voxels.shape[1], self.data_voxels.shape[2],
                                                         self.data_voxels.shape[3]])

        ### get file
        label_txt_path = data_path[:-5] + '.txt'
        self.obj_paths = [os.path.basename(line).rstrip('\n') for line in open(label_txt_path, mode='r').readlines()]

        ## interval
        self.interval = interval


    def __len__(self):
        return self.data_points.shape[0] // self.interval

    def __getitem__(self, idx):

        idx = idx * self.interval

        processed_inputs = self.data_voxels[idx].astype(np.float32), self.data_points[idx].astype(np.float32), \
                           self.data_values[idx]


        return processed_inputs, idx
