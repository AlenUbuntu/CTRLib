import numpy as np 
import torch 
import torch.nn.functional as F


class SparseFeatureLinear(torch.nn.Module):
    def __init__(self, field_info, output_dim=1):
        super(SparseFeatureLinear, self).__init__()

        # call dataset.get_field_info() to get field_info
        self.input_dims = {}
        model_dict = {}
        for ftype in field_info:
            if ftype == 'continuous':
                dims = field_info[ftype]
                model_dict[ftype] = torch.nn.Embedding(
                    num_embeddings=dims,
                    embedding_dim=output_dim,
                )
            elif ftype == 'category':
                dims = field_info[ftype][-1]
                model_dict[ftype] = torch.nn.Embedding(
                    num_embeddings=dims,
                    embedding_dim=output_dim
                )       
            else:
                raise ValueError("Invalid field type {} detected!".format(ftype))
        
            self.input_dims[ftype] = dims
        self.layer = torch.nn.ModuleDict(model_dict)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))

    def forward(self, data):
        output = None
        for ftype in data:
            if ftype == 'continuous':
                assert len(data[ftype].shape) >= 2 and self.input_dims[ftype] == data[ftype].shape[1]
                n = data[ftype].shape[0]
                idx = torch.arange(self.input_dims[ftype]).repeat(n, 1).to(data[ftype].device) # (N, F)
                weights = self.layer[ftype](idx) # (N, F, D)
                x = data[ftype].unsqueeze(-1) # (N, F, 1)
                out = x * weights + self.bias # (N, F, D)
                if output is None:
                    output = out.sum(1)  # (N, D)
                else:
                    output += out.sum(1)
            else:
                assert len(data[ftype].shape) >= 2 and self.input_dims[ftype] == data[ftype].shape[1]
                n = data[ftype].shape[0] # (N, F)
                outs = []
                for i in range(n):
                    x = data[ftype][i] # (F, )
                    indices = torch.nonzero(x, as_tuple=False).to(x.device) # (k, 1)
                    weights = self.layer[ftype](indices) + self.bias # (k, 1, D)
                    out = weights.sum(0) # (1, D)
                    outs.append(out)
                outs = torch.cat(outs, dim=0) # (N, D)
                if output is None:
                    output = outs # (N, D)
                else:
                    output += outs
        return output  # (N, D)


class FactorizationMachineLayer(torch.nn.Module):
    def __init__(self, field_info, emb_dim):
        super(FactorizationMachineLayer, self).__init__()

        # call dataset.get_field_info() to get field_info
        self.input_dims = {}
        model_dict = {}
        for ftype in field_info:
            if ftype == 'continuous':
                dims = field_info[ftype]
                model_dict[ftype] = torch.nn.Embedding(
                    num_embeddings=dims,
                    embedding_dim=emb_dim,
                )
            elif ftype == 'category':
                dims = field_info[ftype][-1]
                model_dict[ftype] = torch.nn.Embedding(
                    num_embeddings=dims,
                    embedding_dim=emb_dim
                )       
            else:
                raise ValueError("Invalid field type {} detected!".format(ftype))
        
            self.input_dims[ftype] = dims
        self.layer = torch.nn.ModuleDict(model_dict)
    
    def forward(self, data):
        assert len(data['continuous'].shape) >= 2 and self.input_dims['continuous'] == data['continuous'].shape[1]
        n = data['continuous'].shape[0]
        idx = torch.arange(self.input_dims['continuous']).repeat(n, 1).to(data['continuous'].device) # (N, F)
        weights = self.layer['continuous'](idx) # (N, F, D)
        x = data['continuous'].unsqueeze(-1) # (N, F, 1)
        output = x * weights # (N, F, D)
        
        assert len(data['category'].shape) >= 2 and self.input_dims['category'] == data['category'].shape[1]
        n = data['category'].shape[0] # (N, F)

        res = [] 
        for i in range(n):
            x = data['category'][i] # (F, )
            indices = torch.nonzero(x, as_tuple=False).to(x.device).squeeze() # (k, )
            out = self.layer['category'](indices) # (k, D), weight * 1

            feats = torch.cat((output[i], out), dim=0) # (F+k, D)
            interaction_mat = torch.matmul(feats, feats.transpose(0, 1)) # (F+k, F+k)
            interaction = torch.sum(interaction_mat) * 0.5
            res.append(interaction.unsqueeze(0))
        return torch.cat(res, dim=0).unsqueeze(dim=1) # (N, 1)
