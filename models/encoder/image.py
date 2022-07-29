import torch
import torch.nn as nn
import torch.nn.functional as F


class resnet_block(nn.Module):
	def __init__(self, dim_in, dim_out):
		super(resnet_block, self).__init__()
		self.dim_in = dim_in
		self.dim_out = dim_out
		if self.dim_in == self.dim_out:
			self.conv_1 = nn.Conv2d(self.dim_in, self.dim_out, 3, stride=1, padding=1, bias=False)
			self.conv_2 = nn.Conv2d(self.dim_out, self.dim_out, 3, stride=1, padding=1, bias=False)
			nn.init.xavier_uniform_(self.conv_1.weight)
			nn.init.xavier_uniform_(self.conv_2.weight)
		else:
			self.conv_1 = nn.Conv2d(self.dim_in, self.dim_out, 3, stride=2, padding=1, bias=False)
			self.conv_2 = nn.Conv2d(self.dim_out, self.dim_out, 3, stride=1, padding=1, bias=False)
			self.conv_s = nn.Conv2d(self.dim_in, self.dim_out, 1, stride=2, padding=0, bias=False)
			nn.init.xavier_uniform_(self.conv_1.weight)
			nn.init.xavier_uniform_(self.conv_2.weight)
			nn.init.xavier_uniform_(self.conv_s.weight)

	def forward(self, input, is_training=False):
		if self.dim_in == self.dim_out:
			output = self.conv_1(input)
			output = F.leaky_relu(output, negative_slope=0.01, inplace=True)
			output = self.conv_2(output)
			output = output+input
			output = F.leaky_relu(output, negative_slope=0.01, inplace=True)
		else:
			output = self.conv_1(input)
			output = F.leaky_relu(output, negative_slope=0.01, inplace=True)
			output = self.conv_2(output)
			input_ = self.conv_s(input)
			output = output+input_
			output = F.leaky_relu(output, negative_slope=0.01, inplace=True)
		return output


class ImageEncoder(nn.Module):
    def __init__(self, config):
        super(ImageEncoder, self).__init__()
        self.img_ef_dim = config.img_ef_dim
        self.z_dim = config.decoder_input_embbeding_size * 2 if config.decoder_type == 'Flow' and hasattr(config,
                                                                                                          'flow_use_split_dim') \
                                                                and config.flow_use_split_dim else config.decoder_input_embbeding_size
        self.conv_0 = nn.Conv2d(1, self.img_ef_dim, 7, stride=2, padding=3, bias=False)
        self.res_1 = resnet_block(self.img_ef_dim, self.img_ef_dim)
        self.res_2 = resnet_block(self.img_ef_dim, self.img_ef_dim)
        self.res_3 = resnet_block(self.img_ef_dim, self.img_ef_dim * 2)
        self.res_4 = resnet_block(self.img_ef_dim * 2, self.img_ef_dim * 2)
        self.res_5 = resnet_block(self.img_ef_dim * 2, self.img_ef_dim * 4)
        self.res_6 = resnet_block(self.img_ef_dim * 4, self.img_ef_dim * 4)
        self.res_7 = resnet_block(self.img_ef_dim * 4, self.img_ef_dim * 8)
        self.res_8 = resnet_block(self.img_ef_dim * 8, self.img_ef_dim * 8)
        self.conv_9 = nn.Conv2d(self.img_ef_dim * 8, self.img_ef_dim * 16, 4, stride=2, padding=1, bias=True)
        self.conv_10 = nn.Conv2d(self.img_ef_dim * 16, self.img_ef_dim * 16, 4, stride=1, padding=0, bias=True)
        self.linear_1 = nn.Linear(self.img_ef_dim * 16, self.img_ef_dim * 16, bias=True)
        self.linear_2 = nn.Linear(self.img_ef_dim * 16, self.img_ef_dim * 16, bias=True)
        self.linear_3 = nn.Linear(self.img_ef_dim * 16, self.img_ef_dim * 16, bias=True)
        self.linear_4 = nn.Linear(self.img_ef_dim * 16, self.z_dim, bias=True)
        nn.init.xavier_uniform_(self.conv_0.weight)
        nn.init.xavier_uniform_(self.conv_9.weight)
        nn.init.constant_(self.conv_9.bias, 0)
        nn.init.xavier_uniform_(self.conv_10.weight)
        nn.init.constant_(self.conv_10.bias, 0)
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.constant_(self.linear_1.bias, 0)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.constant_(self.linear_2.bias, 0)
        nn.init.xavier_uniform_(self.linear_3.weight)
        nn.init.constant_(self.linear_3.bias, 0)
        nn.init.xavier_uniform_(self.linear_4.weight)
        nn.init.constant_(self.linear_4.bias, 0)

    def forward(self, view, is_training=False):
        layer_0 = self.conv_0(1 - view)
        layer_0 = F.leaky_relu(layer_0, negative_slope=0.01, inplace=True)

        layer_1 = self.res_1(layer_0, is_training=is_training)
        layer_2 = self.res_2(layer_1, is_training=is_training)

        layer_3 = self.res_3(layer_2, is_training=is_training)
        layer_4 = self.res_4(layer_3, is_training=is_training)

        layer_5 = self.res_5(layer_4, is_training=is_training)
        layer_6 = self.res_6(layer_5, is_training=is_training)

        layer_7 = self.res_7(layer_6, is_training=is_training)
        layer_8 = self.res_8(layer_7, is_training=is_training)

        layer_9 = self.conv_9(layer_8)
        layer_9 = F.leaky_relu(layer_9, negative_slope=0.01, inplace=True)

        layer_10 = self.conv_10(layer_9)
        layer_10 = layer_10.view(-1, self.img_ef_dim * 16)
        layer_10 = F.leaky_relu(layer_10, negative_slope=0.01, inplace=True)

        l1 = self.linear_1(layer_10)
        l1 = F.leaky_relu(l1, negative_slope=0.01, inplace=True)

        l2 = self.linear_2(l1)
        l2 = F.leaky_relu(l2, negative_slope=0.01, inplace=True)

        l3 = self.linear_3(l2)
        l3 = F.leaky_relu(l3, negative_slope=0.01, inplace=True)

        l4 = self.linear_4(l3)
        l4 = torch.sigmoid(l4)

        return l4