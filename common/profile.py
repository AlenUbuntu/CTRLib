"""
To compute MACs, we only care multiplication and add operations.
"""

from common.efficiency_utils import * 

from models.lr import *
from models.fm import *
from models.dnn import *
from common.layers import *

def prRed(skk): print("\033[91m{}\033[00m".format(skk))


def prGreen(skk): print("\033[92m{}\033[00m".format(skk))


def prYellow(skk): print("\033[93m{}\033[00m".format(skk))


register_hooks = {
    nn.ZeroPad2d: zero_ops,  # padding does not involve any multiplication.

    nn.Conv1d: count_convNd,
    nn.Conv2d: count_convNd,
    nn.Conv3d: count_convNd,
    nn.ConvTranspose1d: count_convNd,
    nn.ConvTranspose2d: count_convNd,
    nn.ConvTranspose3d: count_convNd,

    nn.BatchNorm1d: count_bn,
    nn.BatchNorm2d: count_bn,
    nn.BatchNorm3d: count_bn,

    nn.ReLU: zero_ops,
    nn.ReLU6: zero_ops,
    nn.LeakyReLU: count_relu,

    nn.MaxPool1d: zero_ops,
    nn.MaxPool2d: zero_ops,
    nn.MaxPool3d: zero_ops,
    nn.AdaptiveMaxPool1d: zero_ops,
    nn.AdaptiveMaxPool2d: zero_ops,
    nn.AdaptiveMaxPool3d: zero_ops,

    nn.AvgPool1d: count_avgpool,
    nn.AvgPool2d: count_avgpool,
    nn.AvgPool3d: count_avgpool,
    nn.AdaptiveAvgPool1d: count_adap_avgpool,
    nn.AdaptiveAvgPool2d: count_adap_avgpool,
    nn.AdaptiveAvgPool3d: count_adap_avgpool,

    nn.Linear: count_linear,
    nn.Dropout: zero_ops,

    nn.Upsample: count_upsample,
    nn.UpsamplingBilinear2d: count_upsample,
    nn.UpsamplingNearest2d: count_upsample,

    torch.nn.modules.sparse.Embedding: zero_ops,
    InputLayer: zero_ops,
    SparseFeatureLinear: count_feature_linear,
    MultiLayerPerceptron: count_mlp,

    LogisticRegressionModel: count_lr,
    FactorizationMachineModel: count_fm,
    DNNYouTubeModel: count_dnn,
}


def profile(model: nn.Module, inputs, custom_ops=None, verbose=True):
    fn = None
    m_type = type(model)    

    if custom_ops is None:
        custom_ops = {}

    if m_type in custom_ops:  # if defined both op maps, use custom_ops to overwrite.
        fn = custom_ops[m_type]
        print("[INFO] Customize rule %s() %s." % (fn.__qualname__, m_type))
    elif m_type in register_hooks:
        fn = register_hooks[m_type]
        print("[INFO] Register %s() for %s." % (fn.__qualname__, m_type))
    else:
        prRed("[WARN] Cannot find rule for %s. Treat it as zero Macs and zero Params." % m_type)

    total_ops = register_hooks[m_type](model, *inputs)
    total_params = count_parameters(model, *inputs)

    return total_ops, total_params