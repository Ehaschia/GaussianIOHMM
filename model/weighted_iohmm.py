from typing import Tuple
import numpy as np

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class WeightIOHMM(nn.Module):

    def __init__(self, vocab_size, nlabel, num_state=10):
        super(WeightIOHMM, self).__init__()
        self.vocab_size = vocab_size
        self.num_state = num_state
        self.nlabel = nlabel
        self.input = nn.Embedding(self.vocab_size, self.num_state)
        self.transition = Parameter(torch.empty(self.num_state, self.num_state), requires_grad=True)
        self.output = Parameter(torch.empty(self.num_state, nlabel), requires_grad=True)
        self.criterion = nn.CrossEntropyLoss(reduction='none')

        self.reset_parameter()

    def reset_parameter(self):
        nn.init.uniform_(self.transition.data, a=-0.5, b=0.5)
        nn.init.uniform_(self.input.weight.data, a=0.5, b=0.5)
        nn.init.uniform_(self.output.data, a=-0.5, b=0.5)

    def forward(self, sentences: torch.Tensor) -> torch.Tensor:
        batch, maxlen = sentences.size()

        swapped_sentence = sentences.transpose(0, 1)
        forward_emission = self.input(swapped_sentence.reshape(-1)).reshape(maxlen, batch, self.num_state)
        forward_transition = self.transition.unsqueeze(0)
        # forward
        forwards = [forward_emission[0]]
        for i in range(1, maxlen):
            pre_forward = forwards[i - 1]
            current_forward = torch.matmul(forward_transition, pre_forward.unsqueeze(-1)).squeeze(-1)
            forwards.append(current_forward * forward_emission[i])

        # backward
        backward_emission = forward_emission.flip(dims=[0])
        backward_transition = self.transition.transpose(0, 1).unsqueeze(0)
        backwards = [backward_emission[0]]

        for i in range(1, maxlen):
            pre_backward = backwards[i - 1]
            current_backward = torch.matmul(backward_transition, pre_backward.unsqueeze(-1)).squeeze(-1)
            backwards.append(current_backward * backward_emission[i])

        forwards = torch.stack(forwards, dim=0)
        backwards = torch.stack(backwards[::-1], dim=0)

        expected_count = forwards * backwards / forward_emission

        score = torch.matmul(expected_count, self.output.unsqueeze(0))

        return score.transpose(0, 1)

    def get_loss(self, sentences: torch.Tensor, labels: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        score = self.forward(sentences)
        prob = self.criterion(score.reshape(-1, self.nlabel), labels.view(-1)).reshape_as(labels) * masks
        return torch.sum(prob) / sentences.size(0)

    def get_acc(self, sentences: torch.Tensor, labels: torch.Tensor,
                masks: torch.Tensor) -> Tuple:
        real_score = self.forward(sentences)
        pred = torch.argmax(real_score, dim=-1).cpu().numpy()
        corr = np.sum(np.equal(pred, labels.cpu().numpy()) * masks.cpu().numpy())
        total = np.sum(masks.cpu().numpy())
        return corr / total, corr
