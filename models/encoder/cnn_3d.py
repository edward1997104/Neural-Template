import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN3DDouble(nn.Module):
    def __init__(self, config):
        super(CNN3DDouble, self).__init__()
        self.encoder_1 = CNN3D(config=config)
        self.encoder_2 = CNN3D(config=config)

    def forward(self, inputs):
        embedding_1 = self.encoder_1(inputs)
        embedding_2 = self.encoder_2(inputs)

        return torch.cat((embedding_1, embedding_2), dim=1)


class CNN3D(nn.Module):
    def __init__(self, config):
        super(CNN3D, self).__init__()

        output_dim = config.decoder_input_embbeding_size * 2 if config.decoder_type == 'Flow' and hasattr(config,
                                                                                                          'flow_use_split_dim') \
                                                                and config.flow_use_split_dim else config.decoder_input_embbeding_size

        if hasattr(config, 'use_double_encoder') and config.use_double_encoder:
            output_dim = output_dim // 2

        output_dim = output_dim + config.decoder_input_embbeding_size if hasattr(config,
                                                                                 'bsp_use_binary_prediciton') and config.bsp_use_binary_prediciton else output_dim

        self.ef_dim = int(output_dim / 8)
        self.conv_1 = nn.Conv3d(1, self.ef_dim, 4, stride=2, padding=1, bias=True)
        self.conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim * 2, 4, stride=2, padding=1, bias=True)
        self.conv_3 = nn.Conv3d(self.ef_dim * 2, self.ef_dim * 4, 4, stride=2, padding=1, bias=True)
        self.conv_4 = nn.Conv3d(self.ef_dim * 4, self.ef_dim * 8, 4, stride=2, padding=1, bias=True)
        self.conv_5 = nn.Conv3d(self.ef_dim * 8, self.ef_dim * 8, 4, stride=1, padding=0, bias=True)
        nn.init.xavier_uniform_(self.conv_1.weight)
        nn.init.constant_(self.conv_1.bias, 0)
        nn.init.xavier_uniform_(self.conv_2.weight)
        nn.init.constant_(self.conv_2.bias, 0)
        nn.init.xavier_uniform_(self.conv_3.weight)
        nn.init.constant_(self.conv_3.bias, 0)
        nn.init.xavier_uniform_(self.conv_4.weight)
        nn.init.constant_(self.conv_4.bias, 0)
        nn.init.xavier_uniform_(self.conv_5.weight)
        nn.init.constant_(self.conv_5.bias, 0)


        self.output_dim = output_dim


        self.use_separte_mlp_transform = config.use_separte_mlp_transform if hasattr(config, 'use_separte_mlp_transform') and config.use_separte_mlp_transform else False
        if self.use_separte_mlp_transform:
            self.linear_1 = torch.nn.Sequential(
                torch.nn.Linear(self.ef_dim * 4 , self.ef_dim * 8),
                torch.nn.LeakyReLU(negative_slope= 0.01),
                torch.nn.Linear(self.ef_dim * 8, self.ef_dim * 4)
            )
            self.linear_2 = torch.nn.Sequential(
                torch.nn.Linear(self.ef_dim * 4 , self.ef_dim * 8),
                torch.nn.LeakyReLU(negative_slope= 0.01),
                torch.nn.Linear(self.ef_dim * 8, self.ef_dim * 4)
            )

    def forward(self, inputs, is_training=False):
        d_1 = self.conv_1(inputs)
        d_1 = F.leaky_relu(d_1, negative_slope=0.01, inplace=True)

        d_2 = self.conv_2(d_1)
        d_2 = F.leaky_relu(d_2, negative_slope=0.01, inplace=True)

        d_3 = self.conv_3(d_2)
        d_3 = F.leaky_relu(d_3, negative_slope=0.01, inplace=True)

        d_4 = self.conv_4(d_3)
        d_4 = F.leaky_relu(d_4, negative_slope=0.01, inplace=True)

        d_5 = self.conv_5(d_4)
        d_5 = d_5.view(-1, self.ef_dim * 8)

        if self.use_separte_mlp_transform:
            d_5_1 = self.linear_1(d_5[:, :self.output_dim // 2])
            d_5_2 = self.linear_2(d_5[:, :self.output_dim // 2])
            d_5 = torch.cat((d_5_1, d_5_2), dim = 1)

        d_5 = torch.sigmoid(d_5)

        return d_5