import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import importlib
import os
from models.network import AutoEncoder
from data.data import ImNetSamples
from utils.debugger import MyDebugger
from torch.multiprocessing import Pool, Process, set_start_method

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def extract_one_input(args):
    args_list, network_path, config_path, device_id = args

    torch.cuda.set_device(device_id)
    spec = importlib.util.spec_from_file_location('*', config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    network = AutoEncoder(config=config).cuda(device_id)

    network_state_dict = torch.load(network_path)
    for key, item in list(network_state_dict.items()):
        if key[:7] == 'module.':
            network_state_dict[key[7:]] = item
            del network_state_dict[key]
    network.load_state_dict(network_state_dict)
    network.eval()

    for args in args_list:
        voxels_inputs, file_path, resolution, max_batch, space_range, thershold, obj_path, with_surface_point, file_name = args
        voxels_inputs = torch.from_numpy(voxels_inputs).float().cuda(device_id)
        file = open(file_name, mode='a')
        file.write(f"mesh name for {obj_path}")
        print(f"Outputing {obj_path} for {device_id}")
        network.save_bsp_deform(inputs=voxels_inputs, file_path=file_path, resolution=resolution, max_batch=max_batch,
                                space_range=space_range, thershold_1=thershold)


def split(a, n):
    k, m = divmod(len(a), n)
    return [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

if __name__ == '__main__':


    ## folder for testing
    testing_folder = r'.\pretrain\phase_2_model'
    config_path = os.path.join(testing_folder, 'config.py')

    ## import config here
    spec = importlib.util.spec_from_file_location('*', config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    ## dataload
    ### create dataset
    data_path = r'.\data\all_vox256_img\all_vox256_img_test.hdf5'
    testing_flag = True
    if os.path.exists(config.data_path) and not testing_flag:
        data_path = config.data_path

    model_type = f"AutoEncoder-{config.encoder_type}-{config.decoder_type}" if config.network_type == 'AutoEncoder' else f"AutoDecoder-{config.decoder_type}"

    samples = ImNetSamples(data_path=data_path,
                           sample_voxel_size=config.sample_voxel_size)

    ### debugger0.81684809923172
    debugger = MyDebugger(f'Mesh-visualization-{os.path.basename(config.data_folder)}-{model_type}', is_save_print_to_file = False)


    ## loading index
    epoch = 300
    use_phase = True
    phase = 2
    network_path = os.path.join(testing_folder, f'model_epoch{"_" + str(phase) if use_phase else ""}_{epoch}.pth')

    sample_interval = 1
    resolution = 64
    max_batch = 100000
    save_deformed = True
    thershold = 0.01
    with_surface_point = True
    file = open(debugger.file_path('obj_list.txt'), mode='w')

    device_count = torch.cuda.device_count()
    device_ratio = 1
    worker_nums = int(device_count * device_ratio)
    testing_cnt = 20

    file_name = debugger.file_path('obj_list.txt')
    args = [(samples[i][0][0], debugger.file_path(f'testing_{i}.off'), resolution, max_batch, (-0.5, 0.5), thershold,
             samples.obj_paths[i], with_surface_point, file_name) for i in range(len(samples)) if
            i % sample_interval == 0]
    random.shuffle(args)
    args = args[:testing_cnt]
    splited_args = split(args, worker_nums)
    final_args = [(splited_args[i], network_path, config_path, i % device_count) for i in range(worker_nums)]
    set_start_method('spawn')

    for arg in final_args:
        extract_one_input(arg)
