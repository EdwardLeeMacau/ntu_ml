import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.models import Conformer


class AdditiveMarginSoftmaxLoss(nn.Module):
    # https://arxiv.org/pdf/1801.05599.pdf
    # Reference: https://github.com/Leethony/Additive-Margin-Softmax-Loss-Pytorch/blob/master/AdMSLoss.py
    def __init__(self, s=30.0, m=0.4):
        """ Additive Margin softmax loss """
        # Notes: last layer has no bias (not shifting from origin point)
        super(AdditiveMarginSoftmaxLoss, self).__init__()
        self.s = s
        self.m = m

    def forward(self, x, labels):
        """ input shape (N, in_features) """

        assert len(x) == len(labels)
        assert torch.min(labels) >= 0

        numerator = self.s * (torch.diagonal(x.transpose(0, 1)[labels]) - self.m)

        excl = torch.cat([torch.cat((x[i, :y], x[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)

        L = numerator - torch.log(denominator)
        return -torch.mean(L)

class Classifier(nn.Module):
    def __init__(self, d_model=80, dim_feedforward=256, n_head=2, n_speakers=600, dropout=0.1, normalize=False):
        super().__init__()
        self.normalize = normalize

        # Project the dimension of features from that of input into d_model.
        self.projector = nn.Linear(40, d_model)

        """
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
        # self.encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=d_model, dim_feedforward=dim_feedforward, nhead=n_head, dropout=dropout
        # )
        # self.encoder = nn.TransformerEncoder(
        #     self.encoder_layer, num_layers=4
        # )

        # TODO:
        #   Change Transformer to Conformer.
        #   https://arxiv.org/abs/2005.08100
        #
        # torchaudio implements Conformer for predefined model architecture.
        # need to modify dataset, padding and forwarding function.
        #
        # Reference:
        # https://pytorch.org/audio/stable/generated/torchaudio.models.Conformer.html#torchaudio.models.Conformer
        self.encoder = Conformer(
            input_dim=d_model, num_heads=n_head, ffn_dim=dim_feedforward, num_layers=4,
            depthwise_conv_kernel_size=31
        )

        # TODO:
        #   Implement self-attention encoding pooling
        #   https://arxiv.org/pdf/2008.01077v1.pdf
        self.pooling = nn.Linear(d_model, 1)

        # Project the dimension of features from d_model into speaker nums.
        self.fc0 = nn.Linear(d_model, d_model)
        self.fc1 = nn.Linear(d_model, n_speakers)

    def forward(self, mels, length):
        """
        args:
            mels: (batch size, length, 40)

        return:
            out: (batch size, n_speakers)
        """
        # out: (batch size, length, d_model)
        out = self.projector(mels)
        # out: (length, batch size, d_model)
        # out = out.permute(1, 0, 2)

        # The encoder layer expect features in the shape of (length, batch size, d_model).
        out, _ = self.encoder(out, length)
        # out: (batch size, length, d_model)
        # out = out.transpose(0, 1)

        # mean pooling
        # out: (batch, d_model)
        # out = out.mean(dim=1)

        # self-attention encoding pooling
        # out: (batch, d_model)
        weight = self.pooling(out)
        weight = F.softmax(weight, dim=1)
        out = weight * out
        out = out.sum(dim=1)

        # out: (batch, d_model)
        out = self.fc0(out)
        out = F.sigmoid(out)

        # out: (batch, n_speakers)
        if not self.normalize:
            out = self.fc1(out)
            return out

        # See discussion:
        # https://discuss.pytorch.org/t/how-to-do-weight-normalization-in-last-classification-layer/35193/4
        W = self.fc1.weight / torch.norm(self.fc1.weight, dim=1, keepdim=True)
        out = F.normalize(out, dim=1)
        out = torch.mm(W, out.transpose(0, 1)).transpose(0, 1)

        return out

