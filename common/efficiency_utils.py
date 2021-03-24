"""
MACs: Multiply-Add cumulation
FLOPs： Floating Point Operations?

Most of modern hardware architectures uses FMA instructions for operations with tensors.
FMA computes a*x+b as one operation. Roughly GMACs = 0.5 * GFLOPs
"""
import argparse
import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.utils.rnn import PackedSequence


multiply_adds = 1


def count_parameters(m, x, y):
    total_params = 0
    for p in m.parameters():
        total_params += torch.DoubleTensor([p.numel()])
    return total_params.item()


def zero_ops(m, x, y):
    return float(int(0))


def count_convNd(m: _ConvNd, x: torch.Tensor, y: torch.Tensor):
    # x = x[0] # x - (B, *)

    kernel_ops = torch.zeros(m.weight.size()[2:]).numel()  # Kw x Kh
    bias_ops = 1 if m.bias is not None else 0

    # N x Cout x H x W x  (Cin x Kw x Kh + bias)
    total_ops = y.nelement() * (m.in_channels // m.groups * kernel_ops + bias_ops)

    return float(int(total_ops))


def count_bn(m, x: torch.Tensor, y: torch.Tensor):
    # x = x[0]

    nelements = x.numel()
    total_ops = nelements

    if not m.training:
        # subtract, divide, gamma, beta
        total_ops = 2 * nelements

    return float(int(total_ops))


def count_relu(m, x: torch.Tensor, y: torch.Tensor):
    # x = x[0]

    nelements = x.numel()

    return float(int(nelements))


def count_softmax(m, x: torch.Tensor, y: torch.Tensor):
    # x = x[0]

    batch_size, nfeatures = x.size()

    total_exp = nfeatures
    total_add = nfeatures - 1
    total_div = nfeatures
    total_ops = batch_size * (total_exp + total_add + total_div)

    return float(int(total_ops))


def count_avgpool(m, x: torch.Tensor, y: torch.Tensor):
    total_add = torch.prod(torch.Tensor([m.kernel_size]))
    total_div = 1
    kernel_ops = total_add + total_div
    # kernel_ops = 1
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    return float(int(total_ops))


def count_adap_avgpool(m, x: torch.Tensor, y: torch.Tensor):
    kernel = torch.DoubleTensor([*(x[0].shape[2:])]) // torch.DoubleTensor([*(y.shape[2:])])
    total_add = torch.prod(kernel)
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    return float(int(total_ops))


# TODO: verify the accuracy
def count_upsample(m, x, y):
    if m.mode not in ("nearest", "linear", "bilinear", "bicubic",):  # "trilinear"
        logging.warning("mode %s is not implemented yet, take it a zero op" % m.mode)
        return zero_ops(m, x, y)

    if m.mode == "nearest":
        return zero_ops(m, x, y)

    x = x[0]
    if m.mode == "linear":
        total_ops = y.nelement() * 5  # 2 muls + 3 add
    elif m.mode == "bilinear":
        # https://en.wikipedia.org/wiki/Bilinear_interpolation
        total_ops = y.nelement() * 11  # 6 muls + 5 adds
    elif m.mode == "bicubic":
        # https://en.wikipedia.org/wiki/Bicubic_interpolation
        # Product matrix [4x4] x [4x4] x [4x4]
        ops_solve_A = 224  # 128 muls + 96 adds
        ops_solve_p = 35  # 16 muls + 12 adds + 4 muls + 3 adds
        total_ops = y.nelement() * (ops_solve_A + ops_solve_p)
    elif m.mode == "trilinear":
        # https://en.wikipedia.org/wiki/Trilinear_interpolation
        # can viewed as 2 bilinear + 1 linear
        total_ops = y.nelement() * (13 * 2 + 5)

    return float(total_ops)


def count_linear(m, x, y):
    # per output element
    total_mul = m.in_features
    num_elements = y.numel()
    total_ops = total_mul * num_elements

    return int(total_ops)


def count_feature_linear(m, x, y):
    # x - batch
    total_ops = 0
    for ftype in m.input_dims:
        if ftype == 'continuous':
            # out = x * weights + self.bias # (N, F, D)
            # F * D (w * x + bias)
            total_ops += m.input_dims[ftype] * m.output_dim
            # out.sum(1) (N, D)
            total_ops += m.input_dims[ftype]
        elif ftype == 'category':
            # weights = self.layer[ftype](indices) + self.bias # (k, 1, D)
            # out = weights.sum(0) # (1, D)
            # k_average = x[ftype].view(1, -1).sum(dim=1).item()/len(x[ftype])
            # k * D + k
            k = x[ftype].view(1, -1).sum(dim=1).item()/len(x[ftype])
            total_ops += (k * m.output_dim + k)
    total_ops += 1 # add continuous and category contributions together
    return total_ops


def count_factorization_machine_layer(m, x, y):
    total_ops = 0

    # output = x * weights # (N, F, D)
    # F * D
    total_ops += m.input_dims['continuous'] * m.emb_dim
    # interaction_mat = torch.matmul(feats, feats.transpose(0, 1)) # (F+k, F+k)
    # (D + D-1) * (F+k)^2
    k = x['category'].view(1, -1).sum(dim=1).item()/len(x['category'])
    total_ops += (m.emb_dim + m.emb_dim - 1) * (m.input_dims['continuous']+k)**2
    # nteraction = torch.sum(interaction_mat) * 0.5
    # (F+k)^2 + 1
    total_ops += ((m.input_dims['continuous']+k)**2 + 1)
    return total_ops


def count_mlp(m, x, y):
    total_ops = 0

    models = m.children()
    for each_model in models:
        for i in range(len(each_model)):
            layer = each_model[i]
            model_type = type(layer)
            if model_type == torch.nn.Linear:
                total_ops += count_linear(layer, x, y)
            if model_type == torch.nn.BatchNorm1d:
                total_ops += count_bn(layer, x, y)
            if model_type == torch.nn.ReLU:
                total_ops += zero_ops(layer, x, y)
            if model_type == torch.nn.Dropout:
                total_ops += zero_ops(layer, x, y)

    return total_ops

def count_lr(m, x, y):
    return count_feature_linear(m.linear, x, y)


def count_fm(m, x, y):
    total_ops = 0

    total_ops += count_feature_linear(m.linear, x, y)
    total_ops += count_factorization_machine_layer(m.bi_interaction, x, y)

    total_ops += 1

    return total_ops


def count_dnn(m, x, y):
    total_ops = 0

    x = m.input(x)
    x = x[0]
    total_ops += count_mlp(m.mlp, x, y)

    return total_ops