import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
class BSPDecoder(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.p_dim = config.bsp_p_dim
        self.c_dim = config.bsp_c_dim

        concave_layer_weights = torch.zeros((self.c_dim, 1))
        self.concave_layer_weights = nn.Parameter(concave_layer_weights)
        nn.init.normal_(self.concave_layer_weights, mean=1e-5, std=0.02)


        ## encoder
        self.plane_encoder = PlaneEncoder(config)

        convex_layer_weights = torch.zeros((self.p_dim, self.c_dim))
        self.convex_layer_weights = nn.Parameter(convex_layer_weights)
        nn.init.normal_(self.convex_layer_weights, mean=0.0, std=0.02)

    def forward(self, embedding, x):

        ### convex
        exist = None
        convex_layer_weights = self.convex_layer_weights

        ### encoding
        plane_m = self.plane_encoder(embedding[:, :self.config.decoder_input_embbeding_size])

        ## cat
        ones = torch.ones(x.size(0), x.size(1), 1).to(device)
        x = torch.cat((x, ones), dim = 2)
        if not hasattr(self.config, 'bsp_phase') or self.config.bsp_phase == 0:
            ## phase 0
            h1 = torch.matmul(x, plane_m)
            h1 = torch.clamp(h1, min=0)

            # level 2
            h2 = torch.matmul(h1, convex_layer_weights)
            h2 = torch.clamp(1 - h2, min=0, max=1)

            # level 3
            h3 = torch.matmul(h2, self.concave_layer_weights)
            h3 = torch.clamp(h3, min=0, max=1)
        elif self.config.bsp_phase == 1:
            # level 1
            h1 = torch.matmul(x, plane_m)
            h1 = torch.clamp(h1, min=0)

            # level 2
            h2 = torch.matmul(h1, (convex_layer_weights).float())

            h3 = torch.min(h2, dim=2, keepdim=True)[0]
        else:
            raise Exception("unkown phase!")

        return h2, h3, exist, convex_layer_weights

    def extract_plane(self, embedding):
        plane_m = self.plane_encoder(embedding[:, :self.config.decoder_input_embbeding_size])
        return plane_m

    def set_convex_weight(self, weight):
        self.convex_layer_weights = weight



class PlaneEncoder(torch.nn.Module):
    def __init__(self, config):
        super(PlaneEncoder, self).__init__()
        self.ef_dim = config.decoder_input_embbeding_size // 8
        self.p_dim = config.bsp_p_dim
        self.linear_1 = nn.Linear(self.ef_dim * 8, self.ef_dim * 16, bias=True)
        self.linear_2 = nn.Linear(self.ef_dim * 16, self.ef_dim * 32, bias=True)
        self.linear_3 = nn.Linear(self.ef_dim * 32, self.ef_dim * 64, bias=True)
        self.linear_4 = nn.Linear(self.ef_dim * 64, self.p_dim * 4, bias=True)
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.constant_(self.linear_1.bias, 0)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.constant_(self.linear_2.bias, 0)
        nn.init.xavier_uniform_(self.linear_3.weight)
        nn.init.constant_(self.linear_3.bias, 0)
        nn.init.xavier_uniform_(self.linear_4.weight)
        nn.init.constant_(self.linear_4.bias, 0)

    def forward(self, inputs):
        l1 = self.linear_1(inputs)
        l1 = F.leaky_relu(l1, negative_slope=0.01, inplace=True)

        l2 = self.linear_2(l1)
        l2 = F.leaky_relu(l2, negative_slope=0.01, inplace=True)

        l3 = self.linear_3(l2)
        l3 = F.leaky_relu(l3, negative_slope=0.01, inplace=True)

        l4 = self.linear_4(l3)
        l4 = l4.view(-1, 4, self.p_dim)

        return l4
