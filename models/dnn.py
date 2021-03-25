import torch

from common.layers import *


class DNNYouTubeModel(torch.nn.Module):
    """
    A pytorch implementation of YouTube Deep Neural Network Recommender.

    Reference:
        Paul Covington, Deep Neural Networks for YouTube Recommendations, 2016.
    """

    def __init__(self, cfg, field_info):
        super(DNNYouTubeModel, self).__init__()

        self.input = InputLayer(field_info, cfg.DNN.LATENT_DIM, cfg.DNN.AGGREGATE)

        hidden_dims = []
        cur_dim = self.input.output_dim
        print("Dense Representation Dim: ", cur_dim)
        for fractor in cfg.DNN.HIDDEN_DIMS_FRACTOR:
            hidden_dims.append(int(cur_dim * fractor))
            cur_dim = hidden_dims[-1]
        self.mlp = MultiLayerPerceptron(self.input.output_dim, hidden_dims, cfg.DNN.DROPOUT_PROB)
    
    def forward(self, data):
        x = self.input(data)
        return self.mlp(x)