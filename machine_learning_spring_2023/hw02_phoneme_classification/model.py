from typing import Dict

import torch
from torch import nn
from torch.nn import RNN, GRU
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# Reference:
#
# My implementation from previous course (Applied Deep Learning, Fall 2022).
# https://nol.ntu.edu.tw/nol/coursesearch/print_table.php?course_id=922%20U4340&class=&dpt_code=9440&ser_no=32336&semester=111-1&lang=CH
#
# GitHub URL:
# https://github.com/EdwardLeeMacau/ntucsie_adl/blob/master/intent_classification_and_slot_tagging/model.py#L66
class SeqTagger(nn.Module):
    def __init__(self, input_dim, output_dim=41, hidden_layers=1, hidden_dim=256, bidirectional=True, momentum=0.1, dropout=0.5):
        super(SeqTagger, self).__init__()

        self.num_directions = 2 if bidirectional is True else 1
        self.hidden_dim = hidden_dim

        self.rnn = GRU(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=hidden_layers, dropout=dropout, bidirectional=bidirectional,
            batch_first=True
        )
        self.activation = nn.Tanh()
        self.classifier = nn.Linear(self.encoder_output_size, output_dim)

    @property
    def encoder_output_size(self) -> int:
        return self.num_directions * self.hidden_dim

    def forward(self, batch: Dict) -> torch.Tensor:
        # Extract data from batch
        x, length = batch['sequence'], batch['length']

        # Do not need ONNX compatibility. No needed enforce_sorted
        # output shape = [N, T, hidden_size]
        x = pack_padded_sequence(x, length, batch_first=True, enforce_sorted=False)
        x, _ = self.rnn(x)
        x, length = pad_packed_sequence(x, batch_first=True)

        # output shape = [N, T, C]
        x = self.activation(x)
        x = self.classifier(x)

        # output shape = [N, C, T]
        return x.permute((0, 2, 1))
