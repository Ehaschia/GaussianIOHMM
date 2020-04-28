from typing import Tuple
import numpy as np

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


def nan_detection(tensor):
    if torch.sum(torch.isinf(tensor)) != 0:
        return True
    if torch.sum(torch.isnan(tensor)) != 0:
        return True
    return False


# smoothing every list dim by
def smoothing(tensor, alpha=0.01):
    mean = torch.mean(tensor, dim=-1, keepdim=True)
    return (1 - alpha) * tensor + alpha * mean


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
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.reset_parameter()

    def reset_parameter(self):
        # nn.init.uniform_(self.transition.data, a=-0.1, b=1.0)
        # nn.init.uniform_(self.input.weight.data, a=-0.15, b=0.15)
        # nn.init.uniform_(self.output.data, a=-0.15, b=0.15)
        # nn.init.uniform_(self.transition.data, a=0.0, b=0.15)
        # nn.init.uniform_(self.input.weight.data, a=0.0, b=0.15)
        # nn.init.uniform_(self.output.data, a=0.0, b=0.15)
        nn.init.uniform_(self.transition.data, a=-0.5, b=0.5)
        nn.init.uniform_(self.input.weight.data, a=-0.5, b=0.5)
        nn.init.uniform_(self.output.data, a=-0.5, b=0.5)

    # shape of bm [1, dim, dim]
    # shape of bv [batch, dim]
    @staticmethod
    def bmv_log_product(bm, bv):
        return torch.logsumexp(bm + bv.unsqueeze(-2), dim=-1)

    @staticmethod
    def bmm_log_product(bm1, bm2):
        return torch.logsumexp(bm1.unsqueeze(-1) + bm2.unsqueeze(-3), dim=-2)

    def smooth_parameters(self):
        self.input.weight.data = smoothing(self.input.weight.data)
        self.transition.data = smoothing(self.transition.data)
        self.output.data = smoothing(self.output.data)

    def forward(self, sentences: torch.Tensor) -> torch.Tensor:
        batch, maxlen = sentences.size()

        swapped_sentence = sentences.transpose(0, 1)
        forward_emission = self.logsoftmax(self.input(swapped_sentence.reshape(-1)).reshape(maxlen, batch, self.num_state))
        forward_transition = self.logsoftmax(self.transition.unsqueeze(0))
        # forward
        forwards = [forward_emission[0]]
        # forwards = [self.tanh(forward_emission[0])]
        for i in range(1, maxlen):
            pre_forward = forwards[i - 1]
            current_forward = self.bmv_log_product(forward_transition, pre_forward)
            forwards.append(current_forward + forward_emission[i])
            # current_forward = torch.matmul(forward_transition, pre_forward.unsqueeze(-1)).squeeze(-1)
            # forwards.append(current_forward * forward_emission[i])

        # backward
        backward_emission = forward_emission.flip(dims=[0])
        backward_transition = forward_transition.transpose(1, 2)
        backwards = [backward_emission[0]]

        for i in range(1, maxlen):
            pre_backward = backwards[i - 1]
            current_backward = self.bmv_log_product(backward_transition, pre_backward)
            backwards.append(current_backward + backward_emission[i])
            # current_backward = torch.matmul(backward_transition, pre_backward.unsqueeze(-1)).squeeze(-1)
            # backwards.append(current_backward * backward_emission[i])

        forwards = torch.stack(forwards, dim=0)
        backwards = torch.stack(backwards[::-1], dim=0)
        # nan_detection(forwards)
        # nan_detection(backwards)

        expected_count = forwards + backwards - forward_emission
        expected_count = expected_count - torch.logsumexp(expected_count, dim=-1, keepdim=True)
        # expected_count_debug = forwards * backwards / forward_emission
        # expected_count1 = backwards[0].unsqueeze(0)
        # expected_count2 = torch.matmul(forward_transition.unsqueeze(0), forwards[:-1].unsqueeze(-1)).squeeze(-1) * backwards[1:]
        # expected_count = torch.cat([expected_count1, expected_count2], dim=0)
        # expected_count = smoothing(expected_count)

        # nan_detection(expected_count)
        score = self.bmm_log_product(expected_count, self.logsoftmax(self.output.unsqueeze(0)))
        # score = torch.matmul(expected_count, self.output.unsqueeze(0))
        # nan_detection(score)
        return score.transpose(0, 1)

    def get_loss(self, sentences: torch.Tensor, labels: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        score = self.forward(sentences)
        # prob = self.criterion(score.reshape(-1, self.nlabel), labels.view(-1)).reshape_as(labels) * masks
        prob = score * nn.functional.one_hot(labels, num_classes=self.nlabel) * masks.unsqueeze(-1)
        return -1.0 * torch.sum(prob) / sentences.size(0)

    def get_acc(self, sentences: torch.Tensor, labels: torch.Tensor,
                masks: torch.Tensor) -> Tuple:
        real_score = self.forward(sentences)
        pred = torch.argmax(real_score, dim=-1).cpu().numpy()
        corr = np.sum(np.equal(pred, labels.cpu().numpy()) * masks.cpu().numpy())
        total = np.sum(masks.cpu().numpy())
        return corr / total, corr

# IOHMM for classification
# aka deterministic hmm
class IOHMMClassification(nn.Module):
    def __init__(self, vocab_size, nlabel, num_state=10):
        super(IOHMMClassification, self).__init__()
        self.vocab_size = vocab_size
        self.num_state = num_state
        self.nlabel = nlabel
        self.input = nn.Embedding(self.vocab_size, self.num_state)
        self.transition = Parameter(torch.empty(self.num_state, self.num_state), requires_grad=True)
        self.output = Parameter(torch.empty(nlabel, self.num_state), requires_grad=True)
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.reset_parameter()

    def reset_parameter(self):
        # nn.init.uniform_(self.transition.data, a=-0.1, b=1.0)
        # nn.init.uniform_(self.input.weight.data, a=-0.15, b=0.15)
        # nn.init.uniform_(self.output.data, a=-0.15, b=0.15)
        # nn.init.uniform_(self.transition.data, a=0.0, b=0.15)
        # nn.init.uniform_(self.input.weight.data, a=0.0, b=0.15)
        # nn.init.uniform_(self.output.data, a=0.0, b=0.15)
        nn.init.uniform_(self.transition.data, a=-0.5, b=0.5)
        nn.init.uniform_(self.input.weight.data, a=-0.5, b=0.5)
        nn.init.uniform_(self.output.data, a=-0.5, b=0.5)

    # shape of bm [1, dim, dim]
    # shape of bv [batch, dim]
    @staticmethod
    def bmv_log_product(bm, bv):
        return torch.logsumexp(bm + bv.unsqueeze(-2), dim=-1)

    @staticmethod
    def bmm_log_product(bm1, bm2):
        return torch.logsumexp(bm1.unsqueeze(-1) + bm2.unsqueeze(-3), dim=-2)

    def smooth_parameters(self):
        self.input.weight.data = smoothing(self.input.weight.data)
        self.transition.data = smoothing(self.transition.data)
        self.output.data = smoothing(self.output.data)

    def forward(self, sentences: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        batch, maxlen = sentences.size()

        swapped_sentence = sentences.transpose(0, 1)
        forward_emission = self.logsoftmax(self.input(swapped_sentence.reshape(-1)).reshape(maxlen, batch, self.num_state))
        forward_transition = self.logsoftmax(self.transition.unsqueeze(0))
        # forward
        forwards = [forward_emission[0]]
        # forwards = [self.tanh(forward_emission[0])]
        for i in range(1, maxlen):
            pre_forward = forwards[i - 1]
            current_forward = self.bmv_log_product(forward_transition, pre_forward)
            forwards.append(current_forward + forward_emission[i])
            # current_forward = torch.matmul(forward_transition, pre_forward.unsqueeze(-1)).squeeze(-1)
            # forwards.append(current_forward * forward_emission[i])
        # shape [batch, dim]
        hidden_states = torch.stack(forwards)[length-1, torch.arange(batch), :]

        # shape [batch, label]
        score = self.bmv_log_product(self.logsoftmax(self.output.unsqueeze(0)), hidden_states)
        # score = torch.matmul(expected_count, self.output.unsqueeze(0))
        # nan_detection(score)
        return score

    def get_loss(self, sentences: torch.Tensor, labels: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        length = torch.sum(masks, dim=-1).long()
        score = self.forward(sentences, length)
        # prob = self.criterion(score.reshape(-1, self.nlabel), labels.view(-1)).reshape_as(labels) * masks
        prob = score * nn.functional.one_hot(labels, num_classes=self.nlabel)
        return -1.0 * torch.sum(prob) / sentences.size(0)

    def get_acc(self, sentences: torch.Tensor, labels: torch.Tensor,
                masks: torch.Tensor) -> Tuple:
        length = torch.sum(masks, dim=-1).long()
        real_score = self.forward(sentences, length)
        pred = torch.argmax(real_score, dim=-1).cpu().numpy()
        corr = np.sum(np.equal(pred, labels.cpu().numpy()))
        return corr, pred