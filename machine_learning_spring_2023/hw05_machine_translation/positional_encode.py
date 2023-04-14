import torch
from torch.nn import functional as F
from matplotlib import pyplot as plt
from fairseq.modules import SinusoidalPositionalEmbedding

# Trace TransformDecoder in fairseq, then find member field
# self.embed_positions, it initialize a sinusoidal position embedding
# instance given positional embedding is fixed (not learnable).
pos = SinusoidalPositionalEmbedding.get_embedding(
    num_embeddings=128, embedding_dim=512, padding_idx=1
)
norm = torch.norm(pos, dim=1)

# similarity = F.cosine_similarity(pos, pos)
numerator = norm.repeat(128, 1)
numerator = torch.maximum(numerator * numerator.transpose(0, 1), torch.tensor(1e-8))
similarity = torch.mm(pos, torch.transpose(pos, 0, 1))
similarity /= numerator

plt.imshow(similarity, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.show()
