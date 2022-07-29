import torch
import torch.nn as nn
from models.decoder.ode_layers.cnf_layer import ODE_Net, ODE_ResNet, ODEfunc
from models.decoder.bsp import BSPDecoder
from torchdiffeq import odeint_adjoint, odeint
import os
import numpy as np

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class FlowDecoder(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.trainable_T = self.config.flow_trainable_T
        self.T = config.flow_T
        self.mean_parms = nn.Parameter(torch.tensor(config.initial_mean), requires_grad = True)

        ### init the things needed

        if hasattr(config, 'flow_use_mutilple_layers') and config.flow_use_mutilple_layers:
            self.ode_layers = torch.nn.ModuleList()
            for i in range(config.flows_layers_cnt):
                if not hasattr(config, 'ODE_networkt_type') or config.ODE_networkt_type == 'ODE_Net':
                    ode_net = ODE_Net(config=config).to(device)
                elif config.ODE_networkt_type == 'ODE_ResNet':
                    ode_net = ODE_ResNet(config=config).to(device)
                else:
                    raise Exception("Unknown ODE net!")
                ode_func = ODEfunc(ode_net=ode_net).to(device)

                self.ode_layers.append(ode_func)
        else:
            if not hasattr(config, 'ODE_networkt_type') or config.ODE_networkt_type == 'ODE_Net':
                ode_net = ODE_Net(config=config).to(device)
            elif config.ODE_networkt_type == 'ODE_ResNet':
                ode_net = ODE_ResNet(config = config).to(device)
            else:
                raise Exception("Unknown ODE net!")

            self.ode_func = ODEfunc(ode_net=ode_net).to(device)

        ##
        if config.flow_trainable_T:
            self.register_parameter("sqrt_end_time",
                                    nn.Parameter(torch.sqrt(torch.tensor(config.flow_T)), requires_grad=True))

        self.linear_layer = nn.Linear(3, 1)

        self.bsp_field = BSPDecoder(config)

        self.last_activation = config.decoder_last_activation


    def forward(self, embedding: torch.Tensor, coordinates: torch.Tensor, adjoint=True):

        if hasattr(self.config, 'flow_use_split_dim') and self.config.flow_use_split_dim:
            embedding_1 = embedding[:, :self.config.decoder_input_embbeding_size]
            embedding_2 = embedding[:, self.config.decoder_input_embbeding_size:]
        else:
            embedding_1 = embedding
            embedding_2 = embedding

        final_state = self.forward_flow(embedding_1, coordinates, adjoint=adjoint)

        final_state = self.extract_output(coordinates, embedding_2, final_state)

        return final_state

    def extract_output(self, coordinates, embedding_2, final_state):
        if hasattr(self.config, "flow_use_bsp_field") and self.config.flow_use_bsp_field:
            final_state = self.bsp_field(embedding_2, final_state)
            final_state = (final_state[0], self.last_activation(final_state[1]), final_state[2], final_state[3])
        else:
            raise Exception("Unknown Network....")
        return final_state

    def arbitrary_flow(self, embedding, coordinates: torch.Tensor, start : float, end : float, adjoint=True, device_id=None):
        expanded_embeding = embedding.unsqueeze(1).repeat((1, coordinates.size(1), 1))

        bias = self.config.flow_b if hasattr(self.config, 'flow_b') else 0.0
        bias = np.random.uniform(-bias, bias)

        intergration_times = torch.tensor([torch.tensor(start).to(device), torch.tensor(end).to(device)], requires_grad=True).to(device)

        states = coordinates, expanded_embeding

        if hasattr(self.config, 'flow_use_mutilple_layers') and self.config.flow_use_mutilple_layers:
            assert self.config.flows_layers_cnt > 0
            for i in range(self.config.flows_layers_cnt):
                if adjoint:
                    states_t = odeint_adjoint(
                        self.ode_layers[i],
                        states,
                        intergration_times,
                        atol=self.config.atol,
                        rtol=self.config.rtol
                    )
                else:
                    states_t = odeint(
                        self.ode_layers[i],
                        states,
                        intergration_times,
                        atol=self.config.atol,
                        rtol=self.config.rtol
                    )
                states = states_t[0][1], states_t[1][1]
        else:
            if adjoint:
                states_t = odeint_adjoint(
                    self.ode_func,
                    states,
                    intergration_times,
                    atol=self.config.atol,
                    rtol=self.config.rtol
                )
            else:
                states_t = odeint(
                    self.ode_func,
                    states,
                    intergration_times,
                    atol=self.config.atol,
                    rtol=self.config.rtol
                )

        final_state = states_t[0][1]

        return final_state

    def forward_flow(self, embedding, coordinates: torch.Tensor, adjoint=True, device_id=None, terminate_time = None):
        expanded_embeding = embedding.unsqueeze(1).repeat((1, coordinates.size(1), 1))

        bias = self.config.flow_b if hasattr(self.config, 'flow_b') else 0.0
        bias = np.random.uniform(-bias, bias)

        if terminate_time is None:
            if self.trainable_T:
                intergration_times = torch.stack(
                    [torch.tensor(0.0).to(device), self.sqrt_end_time * self.sqrt_end_time + bias]).to(device)
            else:
                intergration_times = torch.tensor([torch.tensor(0.0).to(device), self.T + bias], requires_grad=True).to(
                    device)
        else:
            intergration_times = torch.tensor([torch.tensor(0.0).to(device), terminate_time], requires_grad=True).to(device)

        states = coordinates, expanded_embeding

        if hasattr(self.config, 'flow_use_mutilple_layers') and self.config.flow_use_mutilple_layers:
            assert self.config.flows_layers_cnt > 0
            for i in range(self.config.flows_layers_cnt):
                if adjoint:
                    states_t = odeint_adjoint(
                        self.ode_layers[i],
                        states,
                        intergration_times,
                        atol=self.config.atol,
                        rtol=self.config.rtol
                    )
                else:
                    states_t = odeint(
                        self.ode_layers[i],
                        states,
                        intergration_times,
                        atol=self.config.atol,
                        rtol=self.config.rtol
                    )
                states = states_t[0][1], states_t[1][1]
        else:
            if adjoint:
                states_t = odeint_adjoint(
                    self.ode_func,
                    states,
                    intergration_times,
                    atol=self.config.atol,
                    rtol=self.config.rtol
                )
            else:
                states_t = odeint(
                    self.ode_func,
                    states,
                    intergration_times,
                    atol=self.config.atol,
                    rtol=self.config.rtol
                )

        final_state = states_t[0][1]

        return final_state

    def reverse_flow(self, embedding, coordinates: torch.Tensor, adjoint=True, terminate_time = None):
        expanded_embeding = embedding.unsqueeze(1).repeat((1, coordinates.size(1), 1))

        if terminate_time is None:
            if self.trainable_T:
                intergration_times = torch.stack(
                    [self.sqrt_end_time * self.sqrt_end_time, torch.tensor(0.0).to(device), ]).to(device)
            else:
                intergration_times = torch.tensor([self.T, torch.tensor(0.0).to(device)], requires_grad=True).to(device)
        else:
            intergration_times = torch.tensor([self.T, self.T * terminate_time], requires_grad=True).to(device)

        states = coordinates, expanded_embeding

        if hasattr(self.config, 'flow_use_mutilple_layers') and self.config.flow_use_mutilple_layers:
            assert self.config.flows_layers_cnt > 0
            for i in range(self.config.flows_layers_cnt):
                if adjoint:
                    states_t = odeint_adjoint(
                        self.ode_layers[i],
                        states,
                        intergration_times,
                        atol=self.config.atol,
                        rtol=self.config.rtol
                    )
                else:
                    states_t = odeint(
                        self.ode_layers[i],
                        states,
                        intergration_times,
                        atol=self.config.atol,
                        rtol=self.config.rtol
                    )
                states = states_t[0][1], states_t[1][1]
        else:
            if adjoint:
                states_t = odeint_adjoint(
                    self.ode_func,
                    states,
                    intergration_times,
                    atol=self.config.atol,
                    rtol=self.config.rtol
                )
            else:
                states_t = odeint_adjoint(
                    self.ode_func,
                    states,
                    intergration_times,
                    atol=self.config.atol,
                    rtol=self.config.rtol
                )

        final_state = states_t[0][1]

        return final_state

if __name__ == '__main__':
    pass
