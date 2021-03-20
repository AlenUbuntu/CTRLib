import numpy as np 
import torch 
import torch.nn.functional as F


class EmbeddingLayer(torch.nn.Module):

    def __init__(self, field_dim, embed_dim, padding_idx=None, act='relu', aggregate=None):
        super(EmbeddingLayer, self).__init__()
        self.embedding = torch.nn.Embedding(field_dim, embed_dim, padding_idx)
        self.embed_dim = embed_dim 

        if aggregate is not None:
            assert aggregate in ('sum', 'avg'), 'EmbeddingLayer: invalid aggregate method detected: {}. Only `sum` and `avg` are supported!'.format(aggregate)
        
        self.aggregate =aggregate 

        # inititalize the embedding
        torch.nn.init.kaiming_normal_(self.embedding.weight)
    
    def forward(self, indices):
        """
        :param indices: One-hot encoding of shape (batch_size, num_embeddings)
        """
        outs = []
        # find embedding of each sample
        for i in range(len(indices)):
            non_zero_indices = torch.nonzero(indices[i], as_tuple=False)

            if len(non_zero_indices) > 0:
                embeddings = self.embedding(non_zero_indices)

                if self.aggregate is None:
                    assert len(embeddings) == 1, "EmbeddingLayer: aggregate is set to be None, but multiple indices are detected: {}".format(non_zero_indices)
                    outs.append(embeddings)
                elif self.aggregate == 'sum':
                    outs.append(torch.sum(embeddings, dim=0))
                elif self.aggregate == 'avg':
                    outs.append(torch.sum(embeddings, dim=0)/len(embeddings))
            else:
                outs.append(torch.zeros(1, self.embed_dim))  # adding zero embedding
        outs = torch.cat(outs, dim=0)

        return outs
        