import torch 

from common.layers import * 

class LogisticRegressionModel(torch.nn.Module):
    """
    A pytorch implementation of Logistic Regression.
    """
    def __init__(self, cfg, field_info):
        super(LogisticRegressionModel, self).__init__()
        self.cfg = cfg 

        # build layers 
        self.linear = SparseFeatureLinear(field_info, output_dim=1)

    def forward(self, data):
        out = self.linear(data)
        return out