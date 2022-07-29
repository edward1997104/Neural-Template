import torch
import torch.nn as nn
import os
from models.decoder.ode_layers import diffeq_layers
class ODE_Net(torch.nn.Module):
    def __init__(self, config ):
        super().__init__()
        base_layer = {
            "ignore": diffeq_layers.IgnoreLinear,
            "squash": diffeq_layers.SquashLinear,
            "scale": diffeq_layers.ScaleLinear,
            "concat": diffeq_layers.ConcatLinear,
            "concat_v2": diffeq_layers.ConcatLinear_v2,
            "concatsquash": diffeq_layers.ConcatSquashLinear,
            "concatscale": diffeq_layers.ConcatScaleLinear,
        }[config.flow_layer_type]

        # build models and add them
        layers = []
        current_dim = config.flow_input_dim

        for layer_dim in (config.flow_layers_dim + [config.flow_input_dim]):
            layer = base_layer(current_dim, layer_dim, config.decoder_input_embbeding_size)
            layers.append(layer)
            current_dim = layer_dim


        self.layers = nn.ModuleList(layers)
        self.activation = config.flow_activation
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, context, x):

        for layer in self.layers[:-1]:
            x = layer(context, x)
            x = self.relu(x)
            ## might change to relu

        ## last layer
        x = self.layers[-1](context, x)

        ## tanh
        x = self.tanh(x)
        return x

class ODE_ResNet(torch.nn.Module):

    ## From neural mesh flow
    def __init__(self, config):
        super().__init__()
        base_layer = {
            "ignore": diffeq_layers.IgnoreLinear,
            "squash": diffeq_layers.SquashLinear,
            "scale": diffeq_layers.ScaleLinear,
            "concat": diffeq_layers.ConcatLinear,
            "concat_v2": diffeq_layers.ConcatLinear_v2,
            "concatsquash": diffeq_layers.ConcatSquashLinear,
            "concatscale": diffeq_layers.ConcatScaleLinear,
        }[config.flow_layer_type]

        assert len(config.flow_layers_dim) > 0
        # build models and add them
        layers_dim = config.flow_layers_dim

        ## first layers
        self.context_linear = nn.Linear(config.decoder_input_embbeding_size + 1, layers_dim[0]) ## include time but might have problem here
        self.coordinate_linear = nn.Linear(config.flow_input_dim, layers_dim[0])

        ## layers
        layers = []
        current_dim = layers_dim[0]
        for next_dim in layers_dim[1:]:
            layer = nn.Linear(current_dim, next_dim)
            nn.init.normal_(layer.weight, mean = 0.0, std = 0.02)
            nn.init.constant_(layer.bias, 0)
            layers.append(layer)
            current_dim = next_dim

        ## last layer
        self.last_linear = nn.Linear(current_dim, config.flow_input_dim)
        nn.init.normal_(self.last_linear.weight, mean = 1e-5, std = 0.02)
        nn.init.constant_(self.last_linear.bias, 0)


        self.layers = nn.ModuleList(layers)
        self.activation = config.flow_activation
        self.relu = nn.Softplus() if hasattr(config,
                                             'flow_resnet_use_softplus') and config.flow_resnet_use_softplus else nn.ReLU()
        self.tanh = nn.Tanh() if not hasattr(config, 'flow_resnet_use_tanh') or config.flow_resnet_use_tanh else lambda \
            x: x

    def forward(self, context, x):

        ## compute embedding for first layer
        x = self.relu(self.coordinate_linear(x))
        context = self.tanh(self.context_linear(context))

        x = x * context

        ## Resnet layer
        for layer in self.layers:
            x = self.relu(layer(x)) + x

        ## matching to output
        dx = self.tanh(self.last_linear(x))

        return dx

class ODEfunc(torch.nn.Module):

    def __init__(self, ode_net : torch.nn.Module):
        super().__init__()
        self.ode_net = ode_net

    def forward(self, t, states):

        y = states[0]
        context = states[1]
        t = t.unsqueeze(0).unsqueeze(1).expand((y.size(0), y.size(1), -1))
        concated_context = torch.cat((t , context), dim = 2)
        dy = self.ode_net(concated_context, y)

        return dy, torch.zeros_like(context).requires_grad_(True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if __name__ == '__main__':

    pass