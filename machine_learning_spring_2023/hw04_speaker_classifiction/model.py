
import torch
import torch.nn as nn
import torch.nn.functional as F


"""# Model
- TransformerEncoderLayer:
  - Base transformer encoder layer in [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
  - Parameters:
    - d_model: the number of expected features of the input (required).

    - nhead: the number of heads of the multi-head attention models (required).

    - dim_feedforward: the dimension of the feed-forward network model (default=2048).

    - dropout: the dropout value (default=0.1).

    - activation: the activation function of intermediate layer, relu or gelu (default=relu).

- TransformerEncoder:
  - TransformerEncoder is a stack of N transformer encoder layers
  - Parameters:
    - encoder_layer: an instance of the TransformerEncoderLayer() class (required).

    - num_layers: the number of sub-encoder-layers in the encoder (required).

    - norm: the layer normalization component (optional).
"""

class Classifier(nn.Module):
	def __init__(self, d_model=80, n_spks=600, dropout=0.1):
		super().__init__()
		# Project the dimension of features from that of input into d_model.
		self.prenet = nn.Linear(40, d_model)
		# TODO:
		#   Change Transformer to Conformer.
		#   https://arxiv.org/abs/2005.08100
		self.encoder_layer = nn.TransformerEncoderLayer(
			d_model=d_model, dim_feedforward=256, nhead=2
		)
		# self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

		# Project the the dimension of features from d_model into speaker nums.
		self.pred_layer = nn.Sequential(
			nn.Linear(d_model, d_model),
			nn.Sigmoid(),
			nn.Linear(d_model, n_spks),
		)

	def forward(self, mels):
		"""
		args:
			mels: (batch size, length, 40)
		return:
			out: (batch size, n_spks)
		"""
		# out: (batch size, length, d_model)
		out = self.prenet(mels)
		# out: (length, batch size, d_model)
		out = out.permute(1, 0, 2)
		# The encoder layer expect features in the shape of (length, batch size, d_model).
		out = self.encoder_layer(out)
		# out: (batch size, length, d_model)
		out = out.transpose(0, 1)
		# mean pooling
		stats = out.mean(dim=1)

		# out: (batch, n_spks)
		out = self.pred_layer(stats)
		return out

