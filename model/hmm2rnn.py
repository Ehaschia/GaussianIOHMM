# this file contains the framework change from HMM to RNN
# like Section 3 of Bridging HMMs and RNNs through Architectural Transformations
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

# According to origin paper, this function not need symbolic root and end.
class HMM(nn.Module):
    def __init__(self, vocab_size, num_state=10):
        super(HMM, self).__init__()
        self.vocab_size = vocab_size
        self.num_state = num_state
        self.input = Parameter(torch.empty(self.vocab_size, self.num_state), requires_grad=True)
        self.transition = Parameter(torch.empty(self.num_state, self.num_state), requires_grad=True)
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.begin = Parameter(torch.empty(self.num_state), requires_grad=True)
        self.logsoftmax1 = nn.LogSoftmax(dim=-1)
        self.logsoftmax0 = nn.LogSoftmax(dim=0)

        self.reset_parameter()

    def reset_parameter(self):
        nn.init.uniform_(self.transition.data, a=-0.5, b=0.5)
        nn.init.uniform_(self.input.data, a=-0.5, b=0.5)
        nn.init.uniform_(self.begin.data, a=-0.5, b=0.5)

    @staticmethod
    def bmv_log_product(bm, bv):
        return torch.logsumexp(bm + bv.unsqueeze(-2), dim=-1)

    @staticmethod
    def bmm_log_product(bm1, bm2):
        return torch.logsumexp(bm1.unsqueeze(-1) + bm2.unsqueeze(-3), dim=-2)

    # log format normalize
    def log_normalize(self, t):
        return self.logsoftmax0(t)

    def forward(self, sentences: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        batch, maxlen = sentences.size()

        # shape [length, batch]
        swapped_sentence = sentences.transpose(0, 1)
        # prob emission (E)
        norm_input = self.logsoftmax0(self.input)

        forward_emission = F.embedding(swapped_sentence.reshape(-1), norm_input).reshape(maxlen, batch, self.num_state)
        # prob transition (W)
        forward_transition = self.logsoftmax1(self.transition.unsqueeze(0))

        # s_0
        forward = self.logsoftmax0(self.begin).expand([batch, self.num_state])

        # forward
        current_mid = self.bmv_log_product(forward_transition, forward)
        mid_forwards = [current_mid]
        for i in range(0, maxlen-1):
            # c_i-1
            pre_forward = mid_forwards[i - 1]
            # s_i
            current_forward = self.log_normalize(pre_forward + forward_emission[i])
            # current_forward = self.bmv_log_product(forward_transition, pre_forward)
            # c_i
            current_mid = self.bmv_log_product(forward_transition, current_forward)
            mid_forwards.append(current_mid)
        # shape [max_len, batch, dim]
        hidden_states = torch.stack(mid_forwards)
        pred_prob = hidden_states + forward_emission
        ppl = torch.logsumexp(pred_prob, dim=-1) * masks.transpose(0, 1)
        return torch.sum(ppl)

    def get_loss(self, sentences: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        ppl = self.forward(sentences, masks)
        return -1.0 * ppl / masks.size(0)
