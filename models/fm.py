import torch

from common.layers import *

class FactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Factorization Machine.

    Reference:
        S Rendle, Factorization Machines, 2010.
    """
    def __init__(self, cfg, field_info):
        super(FactorizationMachineModel, self).__init__()
        self.cfg = cfg 

        # build layers 
        self.linear = SparseFeatureLinear(field_info, output_dim=1)
        self.bi_interaction = FactorizationMachineLayer(field_info, cfg.FM.LATENT_DIM)

    def forward(self, data):
        part1 = self.linear(data) # (N, 1)
        part2 = self.bi_interaction(data) # (N, 1)
        out = part1 + part2 # (N, 1)
        return out