
import torch
import torch.nn as nn
import os
import numpy as np
import torch.nn.functional as F
from models.encoder.dgcnn import DGCNN
from models.encoder.cnn_3d import CNN3D, CNN3DDouble
from models.encoder.image import ImageEncoder
from models.decoder.flow import FlowDecoder
from models.decoder.bsp import BSPDecoder
from utils.ply_utils import triangulate_mesh_with_subdivide
from typing import Union
import mcubes
import math
from utils.other_utils import get_mesh_watertight, write_ply_polygon

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



class ImageAutoEncoder(nn.Module):
    ## init
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_encoder = ImageEncoder(config = self.config)
        self.auto_encoder = None


    def set_autoencoder(self, network):
        self.auto_encoder = network

class ImageAutoEncoderEndToEnd(nn.Module):
    ## init
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_encoder = ImageEncoder(config = self.config)

        network_state_dict = torch.load(self.config.auto_encoder_resume_path)
        network_state_dict = ImageAutoEncoderEndToEnd.process_state_dict(network_state_dict, type = 1)
        self.auto_encoder = AutoEncoder(config)
        self.auto_encoder.load_state_dict(network_state_dict)
        self.auto_encoder.train()

        print(f"Reloaded the Auto encoder from {self.config.auto_encoder_resume_path}")

    def forward(self, images = None, coordinates_inputs = None):

        if coordinates_inputs is None:
            return self.image_encoder(images)
        else:
            embedding = self.image_encoder(images)

            outputs = self.auto_encoder.decoder(embedding, coordinates_inputs)

            return outputs


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




class AutoEncoder(nn.Module):
    ## init
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.encoder_type == 'DGCNN':
            self.encoder = DGCNN(config=config)
        elif config.encoder_type == '3DCNN':
            if hasattr(self.config, 'use_double_encoder') and self.config.use_double_encoder:
                self.encoder = CNN3DDouble(config=config)
            else:
                self.encoder = CNN3D(config=config)

        elif config.encoder_type == 'Image':
            self.encoder = ImageEncoder(config=config)
        else:
            raise Exception("Encoder type not found!")

        if config.decoder_type == 'Flow':
            self.decoder = FlowDecoder(config=config)
        else:
            raise Exception("Decoder type not found!")

    def forward(self, inputs, coordinates_inputs):
        embedding = self.encoder(inputs)
        results = self.decoder(embedding, coordinates_inputs)
        return results


    def create_coordinates(self, resolution, space_range):
        dimensions_samples = np.linspace(space_range[0], space_range[1], resolution)
        x, y, z = np.meshgrid(dimensions_samples, dimensions_samples, dimensions_samples)
        x, y, z = x[:, :, :, np.newaxis], y[:, :, :, np.newaxis], z[:, :, :, np.newaxis]
        coordinates = np.concatenate((x, y, z), axis=3)
        coordinates = coordinates.reshape((-1, 3))
        coordinates = torch.from_numpy(coordinates).float().unsqueeze(0).cuda(device)
        return coordinates


    def save_bsp_deform(self, inputs: torch.Tensor, file_path: Union[None, str],
                        resolution: int = 16, max_batch=100000, space_range=(-1, 1), thershold_1=0.01, thershold_2=0.01, save_output=True, embedding=None):

        assert (self.config.decoder_type == 'Flow' or self.config.decoder_type == 'MVP') and self.config.flow_use_bsp_field

        ## build the coordinates
        coordinates = self.create_coordinates(resolution, space_range)

        ## convex weigth
        convex_layer_weights = self.decoder.bsp_field.convex_layer_weights.detach().cpu().numpy()

        ## get plane
        if embedding is None:
            inputs = inputs.unsqueeze(0)
            embedding = self.encoder(inputs)

        vertices, polygons, vertices_deformed, polygons_deformed, vertices_convex, bsp_convex_list = self.generate_deform_bsp(
            convex_layer_weights, coordinates, embedding, file_path, max_batch,
            resolution, thershold_1, thershold_2, save_output=save_output)

        return vertices, polygons, vertices_deformed, polygons_deformed, embedding, vertices_convex, bsp_convex_list

    def extract_prediction(self, embedding, coordinates, max_batch):

        coordinates = coordinates.unsqueeze(0)
        batch_num = int(np.ceil(coordinates.shape[1] / max_batch))

        results = []
        for i in range(batch_num):
            coordinates_inputs = coordinates[:, i * max_batch:(i + 1) * max_batch]
            result = self.decoder(embedding, coordinates_inputs)[1][0].detach().cpu().numpy()  ## for flow only
            results.append(result)

        if len(results) == 1:
            return results[0]
        else:
            return np.concatenate(tuple(results), axis=0)

    def generate_deform_bsp(self, convex_layer_weights, coordinates, embedding, file_path, max_batch,
                            resolution, thershold_1, thershold_2,
                            save_output=True):

        if hasattr(self.config, 'flow_use_split_dim') and self.config.flow_use_split_dim:
            embedding_1 = embedding[:, :self.config.decoder_input_embbeding_size]
            embedding_2 = embedding[:, self.config.decoder_input_embbeding_size:]
        else:
            embedding_1 = embedding
            embedding_2 = embedding

        bsp_convex_list = self.extract_bsp_convex(convex_layer_weights, coordinates, embedding, max_batch, resolution,
                                                  thershold_1, thershold_2)

        vertices, polygons, vertices_convex, polygons_convex = get_mesh_watertight(bsp_convex_list)

        vertices = np.array(vertices)
        vertices, polygons = triangulate_mesh_with_subdivide(vertices, polygons)

        vertices_result = self.deform_vertices(embedding_1, max_batch, vertices)

        if save_output:
            write_ply_polygon(file_path[:-4] + '_deformed.ply', vertices_result, polygons)
            write_ply_polygon(file_path[:-4] + '_orginal.ply', vertices, polygons)


        return vertices, polygons, vertices_result, polygons, vertices_convex, bsp_convex_list

    def extract_bsp_convex(self, convex_layer_weights, coordinates, embedding, max_batch, resolution, thershold_1,
                           thershold_2):

        if hasattr(self.config, 'flow_use_split_dim') and self.config.flow_use_split_dim:
            embedding_1 = embedding[:, :self.config.decoder_input_embbeding_size]
            embedding_2 = embedding[:, self.config.decoder_input_embbeding_size:]
        else:
            embedding_1 = embedding
            embedding_2 = embedding

        ## plane
        plane_parms = self.decoder.bsp_field.plane_encoder(embedding_2).cpu().detach().numpy()
        convex_predictions = []
        c_dim = self.decoder.bsp_field.c_dim
        for i in range(coordinates.size(1) // max_batch + 1):
            result = self.decoder(embedding, coordinates[:, i * max_batch:(i + 1) * max_batch])
            result = result[0]

            convex_prediction = result.squeeze(0).detach().cpu().numpy()
            convex_predictions.append(convex_prediction)
        if len(convex_predictions) > 1:
            convex_predictions = np.concatenate(tuple(convex_predictions), axis=0)
        else:
            convex_predictions = convex_predictions[0]
        convex_predictions = np.abs(convex_predictions.reshape((resolution, resolution, resolution, c_dim)))
        convex_predictions_float = convex_predictions < thershold_1
        convex_predictions_sum = np.sum(convex_predictions_float, axis=3)
        bsp_convex_list = []
        p_dim = self.decoder.bsp_field.p_dim
        cnt = 0
        for i in range(c_dim):
            slice_i = convex_predictions_float[:, :, :, i]
            if np.max(slice_i) > 0:  # if one voxel is inside a convex
                if np.min(
                        convex_predictions_sum - slice_i * 2) >= 0:  # if this convex is redundant, i.e. the convex is inside the shape
                    convex_predictions_sum = convex_predictions_sum - slice_i
                else:
                    box = []
                    for j in range(p_dim):
                        if convex_layer_weights[j, i] > thershold_2:
                            a = -plane_parms[0, 0, j]
                            b = -plane_parms[0, 1, j]
                            c = -plane_parms[0, 2, j]
                            d = -plane_parms[0, 3, j]
                            box.append([a, b, c, d])
                    if len(box) > 0:
                        bsp_convex_list.append(np.array(box, np.float32))

                cnt += 1
            print(f"{i} done! ")
        print(f'with {len(bsp_convex_list)} convex and enter to function {cnt}')
        return bsp_convex_list

    def deform_vertices(self, embedding, max_batch, vertices, terminate_time = None):
        ### deform the vertices
        vertices_torch = torch.from_numpy(np.array(vertices)).float().to(device).unsqueeze(0)
        vertices_result = []
        for i in range(int(np.ceil(vertices_torch.size(1) / max_batch))):
            result = self.decoder.reverse_flow(embedding, vertices_torch[:, i * max_batch:(i + 1) * max_batch], terminate_time = terminate_time)
            deformed_vertices = result.squeeze(0).detach().cpu().numpy()
            vertices_result.append(deformed_vertices)
        vertices_result = np.concatenate(vertices_result, axis=0)
        return vertices_result

    def undeform_vertices(self, embedding, max_batch, vertices, terminate_time = None):
        ### deform the vertices
        vertices_torch = torch.from_numpy(np.array(vertices)).float().to(device).unsqueeze(0)
        vertices_result = []
        for i in range(int(np.ceil(vertices_torch.size(1) / max_batch))):
            result = self.decoder.forward_flow(embedding, vertices_torch[:, i * max_batch:(i + 1) * max_batch], terminate_time = terminate_time)
            undeformed_vertices = result.squeeze(0).detach().cpu().numpy()
            vertices_result.append(undeformed_vertices)
        vertices_result = np.concatenate(vertices_result, axis=0)
        return vertices_result






if __name__ == '__main__':
    network = None