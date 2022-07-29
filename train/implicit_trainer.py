import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from models.network import AutoEncoder
from data.data import ImNetSamples
from torch.utils.data import DataLoader
from utils.debugger import MyDebugger
import os
import argparse

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# torch.backends.cudnn.benchmark = True

class Trainer(object):

    def __init__(self, config, debugger):
        self.debugger = debugger
        self.config = config

    def train_network(self):

        phases = range(self.config.starting_phase, 3) if hasattr(self.config, 'starting_phase') else [
            int(np.log2(self.config.sample_voxel_size // 16))]

        for phase in phases:
            print(f"Start Phase {phase}")
            sample_voxel_size = 16 * (2 ** (phase))

            if phase == 2:
                if not hasattr(config, 'half_batch_size_when_phase_2') or self.config.half_batch_size_when_phase_2:
                    self.config.batch_size = self.config.batch_size // 2
                self.config.training_epochs = self.config.training_epochs * 2

            ### create dataset
            train_samples = ImNetSamples(data_path=self.config.data_path,
                                         sample_voxel_size=sample_voxel_size)

            train_data_loader = DataLoader(dataset=train_samples,
                                     batch_size=self.config.batch_size,
                                     num_workers=config.data_worker,
                                     shuffle=True,
                                     drop_last=False)

            if hasattr(self.config, 'use_testing') and self.config.use_testing:
                test_samples = ImNetSamples(data_path=self.config.data_path[:-10] + 'test.hdf5',
                                            sample_voxel_size=sample_voxel_size,
                                            interval=self.config.testing_interval)

                test_data_loader = DataLoader(dataset=test_samples,
                                               batch_size=self.config.batch_size,
                                               num_workers=config.data_worker,
                                               shuffle=True,
                                               drop_last=False)

            if self.config.network_type == 'AutoEncoder':
                network = AutoEncoder(config=self.config)
            else:
                raise Exception("Unknown Network type!")

            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                print(f"Use {torch.cuda.device_count()} GPUS!")
                network = nn.DataParallel(network)
            network = network.to(device)

            ## reload the network if needed
            if self.config.network_resume_path is not None:
                network_state_dict = torch.load(self.config.network_resume_path)
                network_state_dict = Trainer.process_state_dict(network_state_dict)
                network.load_state_dict(network_state_dict)
                network.train()
                print(f"Reloaded the network from {self.config.network_resume_path}")
                self.config.network_resume_path = None

            optimizer = torch.optim.Adam(params=network.parameters(), lr=self.config.lr,
                                         betas=(self.config.beta1, 0.999))
            if self.config.optimizer_resume_path is not None:
                optimizer_state_dict = torch.load(self.config.optimizer_resume_path)
                optimizer.load_state_dict(optimizer_state_dict)
                print(f"Reloaded the optimizer from {self.config.optimizer_resume_path}")
                self.config.optimizer_resume_path = None

            for idx in range(self.config.starting_epoch, self.config.training_epochs + 1):
                with tqdm(train_data_loader, unit='batch') as tepoch:
                    tepoch.set_description(f'Epoch {idx}')
                    losses = []
                    losses = self.evaluate_one_epoch(losses, network, optimizer, tepoch, is_training = True)

                    print(f"Test Loss for epoch {idx} : {np.mean(losses)}")


                    ## saving the models
                    if idx % self.config.saving_intervals == 0:
                        # save
                        save_model_path = self.debugger.file_path(f'model_epoch_{phase}_{idx}.pth')
                        save_optimizer_path = self.debugger.file_path(f'optimizer_epoch_{phase}_{idx}.pth')
                        torch.save(network.state_dict(), save_model_path)
                        torch.save(optimizer.state_dict(), save_optimizer_path)
                        print(f"Epoch {idx} model saved at {save_model_path}")
                        print(f"Epoch {idx} optimizer saved at {save_optimizer_path}")
                        self.config.network_resume_path = save_model_path  ## add this resume after the whole things are compelete

                        if hasattr(self.config, 'use_testing') and self.config.use_testing:
                            with tqdm(test_data_loader, unit='batch') as tepoch:
                                losses = []
                                losses = self.evaluate_one_epoch(losses, network, optimizer = None, tepoch = tepoch, is_training=False)
                                print(f"Test Loss for epoch {idx} : {np.mean(losses)}")

            ## when done the phase
            self.config.starting_epoch = 0

    def evaluate_one_epoch(self, losses, network, optimizer, tepoch, is_training = True):
        ## main training loop
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for inputs, samples_indices in tepoch:

            ## get voxel_inputs
            voxels_inputs, coordinate_inputs, occupancy_ground_truth = inputs
            normals_gt = None

            ## remove gradient
            if is_training:
                optimizer.zero_grad()

            voxels_inputs, coordinate_inputs, occupancy_ground_truth, samples_indices = voxels_inputs.to(
                device), coordinate_inputs.to(device), occupancy_ground_truth.to(
                device), samples_indices.to(device)

            if self.config.network_type == 'AutoEncoder':
                prediction = network(voxels_inputs, coordinate_inputs)
            else:
                raise Exception("Unknown Network Type....")

            convex_prediction, prediction, exist, convex_layer_weights = self.extract_prediction(prediction)

            loss = self.config.loss_fn(torch.clamp(prediction, min=0, max=1), occupancy_ground_truth)

            ### loss function to be refactor
            if (self.config.decoder_type == 'Flow' and self.config.flow_use_bsp_field == True) or self.config.decoder_type == 'MVP':
                loss, losses = self.flow_bsp_loss(loss, losses, network,
                                                  occupancy_ground_truth,
                                                  prediction, convex_layer_weights)
            else:
                raise Exception("Unknown Network Type....")

            if is_training:
                loss.backward()
                optimizer.step()

            tepoch.set_postfix(loss=f'{np.mean(losses)}')

        return losses


    def flow_bsp_loss(self, loss, losses, network, occupancy_ground_truth, prediction, convex_layer_weights):

        bsp_thershold = self.config.bsp_thershold if hasattr(self.config, 'bsp_thershold') else 0.01
        if self.config.bsp_phase == 0:
            concave_layer_weights = network.decoder.bsp_field.concave_layer_weights if torch.cuda.device_count() <= 1 else network.module.decoder.bsp_field.concave_layer_weights
            losses.append(loss.detach().item())
            loss = loss + torch.sum(
                torch.abs(concave_layer_weights - 1))  ### convex layer weight close to 1
            loss = loss + torch.sum(
                torch.clamp(convex_layer_weights - 1, min=0) - torch.clamp(convex_layer_weights,
                                                                           max=0))
        elif self.config.bsp_phase == 1:
            loss = torch.mean((1 - occupancy_ground_truth) * (
                    1 - torch.clamp(prediction, max=1)) + occupancy_ground_truth * torch.clamp(
                prediction, min=0))
            losses.append(loss.detach().item())
            loss = loss + torch.sum(
                (convex_layer_weights < bsp_thershold).float() * torch.abs(
                    convex_layer_weights)) + torch.sum(
                (convex_layer_weights >= bsp_thershold).float() * torch.abs(convex_layer_weights - 1))
        else:
            raise Exception("Unknown Phase.....")


        return loss, losses


    def extract_prediction(self, prediction):

        assert self.config.decoder_type == 'Flow' and self.config.flow_use_bsp_field == True
        convex_prediction, prediction, exist, convex_layer_weights = prediction  # unpack the idead

        return convex_prediction, prediction, exist, convex_layer_weights

    @staticmethod
    def process_state_dict(network_state_dict, type = 0):

        if torch.cuda.device_count() >= 2 and type == 0:
            for key, item in list(network_state_dict.items()):
                if key[:7] != 'module.':
                    new_key = 'module.' + key
                    network_state_dict[new_key] = item
                    del network_state_dict[key]
        else:
            for key, item in list(network_state_dict.items()):
                if key[:7] == 'module.':
                    new_key = key[7:]
                    network_state_dict[new_key] = item
                    del network_state_dict[key]

        return network_state_dict


if __name__ == '__main__':
    import importlib

    ## additional args for parsing
    optional_args = [("network_resume_path", str), ("optimizer_resume_path", str), ("starting_epoch", int),
                     ("special_symbol", str), ("resume_path", str), ("starting_phase", int)]
    parser = argparse.ArgumentParser()
    for optional_arg, arg_type in optional_args:
        parser.add_argument(f"--{optional_arg}", help=f"{optional_arg}", type=arg_type)

    args = parser.parse_args()
    ## Resume setting
    resume_path = None

    ## resume from path if needed
    if args.resume_path is not None:
        resume_path = args.resume_path

    if resume_path is None:
        from configs import config
        resume_path = os.path.join('configs', 'config.py')
    else:
        ## import config here
        spec = importlib.util.spec_from_file_location('*', resume_path)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)

    for optional_arg, arg_type in optional_args:
        if args.__dict__.get(optional_arg, None) is not None:
            locals()['config'].__setattr__(optional_arg, args.__dict__.get(optional_arg, None))


    model_type = f"AutoEncoder-{config.encoder_type}-{config.decoder_type}" if config.network_type == 'AutoEncoder' else f"AutoDecoder-{config.decoder_type}"
    debugger = MyDebugger(f'IM-Net-Training-experiment-{os.path.basename(config.data_folder)}-{model_type}{config.special_symbol}', is_save_print_to_file = True, config_path = resume_path)
    trainer = Trainer(config = config, debugger = debugger)
    trainer.train_network()
