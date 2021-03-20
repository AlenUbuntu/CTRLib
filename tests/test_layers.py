import random 
import numpy as np
import torch
import unittest
from common.layers import EmbeddingLayer


class TestLayers(unittest.TestCase):

    def test_embedding_layer(self):
        num_embeddings = random.randint(1, 1000)
        embedding_dim = random.randint(1, 1000)
        num_samples = random.randint(1, 100)
        aggregate = random.choice(('sum', 'avg'))
        # print("Number of Embeddings: {} - Embedding Dim: {} - Number of Samples: {} - Aggregate: {}".format(num_embeddings, embedding_dim, num_samples, aggregate))
        one_hot_indices = torch.zeros(num_samples, num_embeddings, dtype=torch.long)
        x = np.random.randint(0, num_samples, size=num_samples * 3)
        y = np.random.randint(0, num_embeddings, size=num_samples * 3)
        for i, j in zip(x, y):
            one_hot_indices[i,j] = 1
        self.assertEqual(one_hot_indices.shape, (num_samples, num_embeddings))
        layer = EmbeddingLayer(num_embeddings, embedding_dim, aggregate=aggregate)
        embedding = layer(one_hot_indices)
        self.assertEqual(embedding.shape, (num_samples, embedding_dim))
        self.assertTrue(((one_hot_indices == 1).any(dim=1) == (embedding != 0).any(dim=1)).all())

if __name__ == '__main__':
    unittest.main()