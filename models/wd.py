
import torch

from common.layers import *


class WideAndDeepModel(torch.nn.Module):
    """
    A pytorch implementation of Wide & Deep model.

    Reference:
        Heng-Tze Cheng, Wide & Deep Learning for Recommender Systems, 2016
    """
    def __init__(self, cfg, field_info, cross_prod_transform=None):
        super(WideAndDeepModel, self).__init__()
        self.linear = SparseFeatureLinear(field_info)
        self.input = InputLayer(field_info, cfg.WD.LATENT_DIM, cfg.WD.AGGREGATE)
        
        hidden_dims = []
        cur_dim = self.input.output_dim
        for fractor in cfg.WD.HIDDEN_DIMS_FRACTOR:
            hidden_dims.append(int(cur_dim * fractor))
            cur_dim = hidden_dims[-1]
        self.mlp = MultiLayerPerceptron(self.input.output_dim, hidden_dims, cfg.WD.DROPOUT_PROB)
        self.bias = torch.nn.Parameter(torch.zeros(1, ))
        self.cross_prod_transform = cross_prod_transform

    def forward(self, data):
        wide_part = self.linear(self.cross_prod_transform(data)) if self.cross_prod_transform else self.linear(data)
        deep_part = self.mlp(self.input(data))
        logit = wide_part + deep_part + self.bias 

        return logit