import torch
import torch.nn as nn

class FlowMLPfield(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        current_dim = config.flow_input_dim
        if config.flow_field_use_embedding_dim:
            current_dim += config.decoder_input_embbeding_size

        self.layers = torch.nn.ModuleList()

        ## layers
        for layer_dim in config.flow_mlp_field_layers:
            layer = nn.Linear(current_dim, layer_dim)
            self.layers.append(layer)
            current_dim = layer_dim


        ## last linear
        self.last_linear = nn.Linear(current_dim, 1)

        ## activation
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.relu(self.layers[0](x))
        for layer in self.layers[1:]:
            x = self.tanh(layer(x)) + x

        x = self.last_linear(x)

        return x