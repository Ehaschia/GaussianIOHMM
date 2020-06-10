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

    def debug_init(self):
        self.transition.data = torch.tensor([[0.7, 0.3], [0.1, 0.9]])
        self.input.data = torch.tensor([[10.0, -1.0], [-1.0, 10.0]])
        self.begin.data = torch.tensor([-1.0, 10.0])

    @staticmethod
    def bmv_log_product(bm, bv):
        return torch.logsumexp(bm + bv.unsqueeze(-2), dim=-1)

    @staticmethod
    def bvm_log_product(bm, bv):
        return torch.logsumexp(bm + bv.unsqueeze(-1), dim=-2)

    @staticmethod
    def bmm_log_product(bm1, bm2):
        return torch.logsumexp(bm1.unsqueeze(-1) + bm2.unsqueeze(-3), dim=-2)

    # log format normalize
    def log_normalize(self, t):
        return self.logsoftmax1(t)

    def forward(self, sentences: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        batch, maxlen = sentences.size()

        # shape [length, batch]
        swapped_sentence = sentences.transpose(0, 1)
        # prob emission (E)
        norm_input = self.logsoftmax0(self.input)

        emission = F.embedding(swapped_sentence.reshape(-1), norm_input).reshape(maxlen, batch, self.num_state)
        # prob transition (W)
        forward_transition = self.logsoftmax1(self.transition.unsqueeze(0))

        # s_0
        current_mid = self.logsoftmax0(self.begin).expand([batch, self.num_state])
        mid_forwards = [current_mid]
        for i in range(0, maxlen-1):
            # c_i-1
            pre_forward = mid_forwards[i]
            # s_i
            current_forward = self.log_normalize(pre_forward + emission[i])
            # c_i
            current_mid = self.bvm_log_product(forward_transition, current_forward)
            # current_mid = torch.logsumexp(forward_transition + current_forward.unsqueeze(-1), dim=-1)
            mid_forwards.append(current_mid)
        # shape [max_len, batch, dim]
        hidden_states = torch.stack(mid_forwards)
        pred_prob = hidden_states + emission
        ppl = torch.logsumexp(pred_prob, dim=-1) * masks.transpose(0, 1)
        return torch.sum(ppl)

    def get_loss(self, sentences: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        ppl = self.forward(sentences, masks)
        return -1.0 * ppl / masks.size(0)

# tensor based
class TBHMM(nn.Module):
    def __init__(self, vocab_size, num_state=10):
        super(TBHMM, self).__init__()
        self.vocab_size = vocab_size
        self.num_state = num_state
        self.input = Parameter(torch.empty(self.vocab_size, self.num_state), requires_grad=True)
        self.transition = Parameter(torch.empty(self.num_state, self.num_state, self.num_state), requires_grad=True)
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

    @staticmethod
    def bvm_log_product(bm, bv):
        return torch.logsumexp(bm + bv.unsqueeze(-1), dim=-2)


    # log format normalize
    def log_normalize(self, t):
        return self.logsoftmax1(t)

    def forward(self, sentences: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        batch, maxlen = sentences.size()

        # shape [length, batch]
        swapped_sentence = sentences.transpose(0, 1)
        # prob emission (E)
        norm_input = self.logsoftmax0(self.input)

        log_e = F.embedding(swapped_sentence.reshape(-1), norm_input).reshape(maxlen, batch, self.num_state)
        # prob transition (W) shape [1, state, state, state]
        forward_transition = self.transition.unsqueeze(0)

        # c_0, shape [batch, state]
        current_mid = self.logsoftmax0(self.begin).expand([batch, self.num_state])

        # forward
        mid_forwards = [current_mid]
        for i in range(0, maxlen-1):
            # c_i-1
            pre_forward = mid_forwards[i - 1]
            # s_i
            current_forward = self.log_normalize(pre_forward + log_e[i])
            # c_i
            current_mid = self.logsoftmax1(torch.matmul(forward_transition, torch.exp(log_e[i].view(batch, 1, self.num_state, 1))).squeeze(-1))
            # TODO find the difference?
            # current_mid = torch.logsumexp(current_mid + current_forward.unsqueeze(-1), dim=-2)
            current_mid = self.bvm_log_product(current_mid, current_forward)
            # current_mid = torch.logsumexp(current_mid + current_forward.unsqueeze(-1), dim=-1)
            mid_forwards.append(current_mid)
        # shape [max_len, batch, dim]
        hidden_states = torch.stack(mid_forwards)
        pred_prob = hidden_states + log_e
        ppl = torch.logsumexp(pred_prob, dim=-1) * masks.transpose(0, 1)
        return torch.sum(ppl)

    def get_loss(self, sentences: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        ppl = self.forward(sentences, masks)
        return -1.0 * ppl / masks.size(0)

# addition based
class ABHMM(nn.Module):
    def __init__(self, vocab_size, num_state=10):
        super(ABHMM, self).__init__()
        self.vocab_size = vocab_size
        self.num_state = num_state
        self.input = Parameter(torch.empty(self.vocab_size, self.num_state), requires_grad=True)
        self.transition = Parameter(torch.empty(self.num_state, self.num_state), requires_grad=True)
        # U
        self.emission_transition = nn.Linear(self.num_state, self.num_state)
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.begin = Parameter(torch.empty(self.num_state), requires_grad=True)
        self.logsoftmax1 = nn.LogSoftmax(dim=-1)
        self.logsoftmax0 = nn.LogSoftmax(dim=0)

        self.reset_parameter()

    def reset_parameter(self):
        nn.init.uniform_(self.transition.data, a=-0.5, b=0.5)
        nn.init.uniform_(self.input.data, a=-0.5, b=0.5)
        nn.init.uniform_(self.begin.data, a=-0.5, b=0.5)
        nn.init.zeros_(self.emission_transition.bias.data)
        nn.init.uniform_(self.emission_transition.weight.data, a=-0.5, b=0.5)


    @staticmethod
    def bmv_log_product(bm, bv):
        return torch.logsumexp(bm + bv.unsqueeze(-2), dim=-1)

    @staticmethod
    def bmm_log_product(bm1, bm2):
        return torch.logsumexp(bm1.unsqueeze(-1) + bm2.unsqueeze(-3), dim=-2)

    @staticmethod
    def bvm_log_product(bm, bv):
        return torch.logsumexp(bm + bv.unsqueeze(-1), dim=-2)


    # log format normalize
    def log_normalize(self, t):
        return self.logsoftmax1(t)

    def forward(self, sentences: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        batch, maxlen = sentences.size()

        # shape [length, batch]
        swapped_sentence = sentences.transpose(0, 1)
        # prob emission (E)
        norm_input = self.logsoftmax0(self.input)

        log_e = F.embedding(swapped_sentence.reshape(-1), norm_input).reshape(maxlen, batch, self.num_state)
        # prob transition (W) shape [1, state, state, state]
        forward_transition = self.transition.unsqueeze(0)

        # c_0, shape [batch, state]
        current_mid = self.logsoftmax0(self.begin).expand([batch, self.num_state])

        # forward
        mid_forwards = [current_mid]
        for i in range(0, maxlen-1):
            # c_i-1
            pre_forward = mid_forwards[i]
            # s_i
            current_forward = self.log_normalize(pre_forward + log_e[i])
            # c_i
            current_mid = self.logsoftmax1(self.emission_transition(torch.exp(log_e[i])).unsqueeze(-2) + forward_transition)
            current_mid = self.bvm_log_product(current_mid, current_forward)
            # current_mid = torch.logsumexp(current_mid + current_forward.unsqueeze(-1), dim=-1)
            mid_forwards.append(current_mid)
        # shape [max_len, batch, dim]
        hidden_states = torch.stack(mid_forwards)
        pred_prob = hidden_states + log_e
        ppl = torch.logsumexp(pred_prob, dim=-1) * masks.transpose(0, 1)
        return torch.sum(ppl)

    def get_loss(self, sentences: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        ppl = self.forward(sentences, masks)
        return -1.0 * ppl / masks.size(0)


# gate based
class GBHMM(nn.Module):
    def __init__(self, vocab_size, num_state=10):
        super(GBHMM, self).__init__()
        self.vocab_size = vocab_size
        self.num_state = num_state
        self.input = Parameter(torch.empty(self.vocab_size, self.num_state), requires_grad=True)
        self.transition = Parameter(torch.empty(self.num_state, self.num_state), requires_grad=True)
        self.emission_gate = nn.Linear(self.num_state, self.num_state)
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.begin = Parameter(torch.empty(self.num_state), requires_grad=True)
        self.logsoftmax1 = nn.LogSoftmax(dim=-1)
        self.logsoftmax0 = nn.LogSoftmax(dim=0)

        self.reset_parameter()

    def reset_parameter(self):
        nn.init.uniform_(self.transition.data, a=-0.5, b=0.5)
        nn.init.uniform_(self.input.data, a=-0.5, b=0.5)
        nn.init.uniform_(self.begin.data, a=-0.5, b=0.5)
        nn.init.zeros_(self.emission_gate.bias.data)
        nn.init.uniform_(self.emission_gate.weight.data, a=-0.5, b=0.5)

    @staticmethod
    def bmv_log_product(bm, bv):
        return torch.logsumexp(bm + bv.unsqueeze(-2), dim=-1)

    @staticmethod
    def bmm_log_product(bm1, bm2):
        return torch.logsumexp(bm1.unsqueeze(-1) + bm2.unsqueeze(-3), dim=-2)

    @staticmethod
    def bvm_log_product(bm, bv):
        return torch.logsumexp(bm + bv.unsqueeze(-1), dim=-2)

    # log format normalize
    def log_normalize(self, t):
        return self.logsoftmax1(t)

    def forward(self, sentences: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        batch, maxlen = sentences.size()

        # shape [length, batch]
        swapped_sentence = sentences.transpose(0, 1)
        # prob emission
        prob_emission = self.logsoftmax0(self.input)
        # emission (E)
        emission = F.embedding(swapped_sentence.reshape(-1), self.input).reshape(maxlen, batch, self.num_state)
        # equation 25
        prob_e = F.embedding(swapped_sentence.reshape(-1), prob_emission).reshape(maxlen, batch, self.num_state)
        fe = torch.sigmoid(self.emission_gate(emission))
        # prob transition (W) shape [1, state, state]
        forward_transition = self.transition.unsqueeze(0)

        # c_0, shape [batch, state]
        current_mid = self.logsoftmax0(self.begin).expand([batch, self.num_state])

        # forward
        mid_forwards = [current_mid]
        for i in range(0, maxlen-1):
            # c_i-1
            pre_forward = mid_forwards[i]
            # s_i
            current_forward = self.log_normalize(pre_forward + prob_e[i])
            # c_i
            current_mid = self.logsoftmax1(forward_transition * fe[i].unsqueeze(1))
            current_mid = self.bvm_log_product(current_mid, current_forward)
            # current_mid = torch.logsumexp(current_mid + current_forward.unsqueeze(-1), dim=-1)
            mid_forwards.append(current_mid)
        # shape [max_len, batch, dim]
        hidden_states = torch.stack(mid_forwards)
        pred_prob = hidden_states + prob_e
        ppl = torch.logsumexp(pred_prob, dim=-1) * masks.transpose(0, 1)
        return torch.sum(ppl)

    def get_loss(self, sentences: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        ppl = self.forward(sentences, masks)
        return -1.0 * ppl / masks.size(0)


# Decomposed based
class DBHMM(nn.Module):
    def __init__(self, vocab_size, num_state1=10, num_state2=10):
        super(DBHMM, self).__init__()
        self.vocab_size = vocab_size
        self.num_state1 = num_state1
        self.num_state2 = num_state2
        self.input = Parameter(torch.empty(self.vocab_size, self.num_state1), requires_grad=True)
        self.d1 = Parameter(torch.empty(self.num_state2, self.num_state1), requires_grad=True)
        self.d2 = Parameter(torch.empty(self.num_state2, self.num_state1), requires_grad=True)
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.begin = Parameter(torch.empty(self.num_state2), requires_grad=True)
        self.logsoftmax1 = nn.LogSoftmax(dim=-1)
        self.logsoftmax0 = nn.LogSoftmax(dim=0)

        self.reset_parameter()

    def reset_parameter(self):
        nn.init.uniform_(self.d1.data, a=-0.1, b=0.1)
        nn.init.uniform_(self.d2.data, a=-0.1, b=0.1)
        nn.init.uniform_(self.input.data, a=-0.1, b=0.1)
        nn.init.uniform_(self.begin.data, a=-0.1, b=0.1)

    @staticmethod
    def bmv_log_product(bm, bv):
        return torch.logsumexp(bm + bv.unsqueeze(-2), dim=-1)

    @staticmethod
    def bmm_log_product(bm1, bm2):
        return torch.logsumexp(bm1.unsqueeze(-1) + bm2.unsqueeze(-3), dim=-2)

    @staticmethod
    def bvm_log_product(bm, bv):
        return torch.logsumexp(bm + bv.unsqueeze(-1), dim=-2)

    # log format normalize
    def log_normalize(self, t):
        return self.logsoftmax1(t)

    def forward(self, sentences: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        batch, maxlen = sentences.size()

        # shape [length, batch]
        swapped_sentence = sentences.transpose(0, 1)
        # prob emission (E)
        norm_input = self.logsoftmax0(self.input)

        log_e = F.embedding(swapped_sentence.reshape(-1), norm_input).reshape(maxlen, batch, self.num_state1)

        # c_0, shape [batch, state]
        current_mid = self.logsoftmax0(self.begin).expand([batch, self.num_state2])

        # forward
        mid_forwards = [current_mid]
        for i in range(0, maxlen-1):
            # c_i-1
            pre_mid = mid_forwards[i - 1]
            # c_i
            a = torch.matmul(pre_mid.unsqueeze(1), self.d1.unsqueeze(0)).squeeze(1)
            current_mid = self.logsoftmax1(torch.matmul(a.unsqueeze(1), self.d2.unsqueeze(0).transpose(1, 2)).squeeze(1))

            mid_forwards.append(current_mid)
        # shape [max_len, batch, dim]
        hidden_states = torch.stack(mid_forwards)
        pred_prob = hidden_states + log_e
        ppl = torch.logsumexp(pred_prob, dim=-1) * masks.transpose(0, 1)
        return torch.sum(ppl)

    def get_loss(self, sentences: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        ppl = self.forward(sentences, masks)
        return -1.0 * ppl / masks.size(0)


# delay transition
class DTHMM(nn.Module):
    def __init__(self, vocab_size, num_state=10):
        super(DTHMM, self).__init__()
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

    @staticmethod
    def bvm_log_product(bm, bv):
        return torch.logsumexp(bm + bv.unsqueeze(-1), dim=-2)


    # log format normalize
    def log_normalize(self, t):
        return self.logsoftmax1(t)

    def forward(self, sentences: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        batch, maxlen = sentences.size()

        # shape [length, batch]
        swapped_sentence = sentences.transpose(0, 1)
        # prob emission (E)
        norm_input = self.logsoftmax0(self.input)

        log_e = F.embedding(swapped_sentence.reshape(-1), norm_input).reshape(maxlen, batch, self.num_state)
        # prob transition (W) shape [1, state, state, state]
        forward_transition = self.transition.unsqueeze(0)

        # c_0, shape [batch, state]
        current_mid = self.logsoftmax0(self.begin).expand([batch, self.num_state])

        # forward
        mid_forwards = [current_mid]
        for i in range(0, maxlen - 1):
            # c_i-1
            pre_forward = mid_forwards[i]
            # s_i
            current_forward = self.log_normalize(pre_forward + log_e[i])
            # c_i
            current_mid = self.logsoftmax1(torch.matmul(forward_transition, torch.exp(current_forward).unsqueeze(-1)).squeeze(-1))
            mid_forwards.append(current_mid)
        # shape [max_len, batch, dim]
        hidden_states = torch.stack(mid_forwards)
        pred_prob = hidden_states + log_e
        ppl = torch.logsumexp(pred_prob, dim=-1) * masks.transpose(0, 1)
        return torch.sum(ppl)

    def get_loss(self, sentences: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        ppl = self.forward(sentences, masks)
        return -1.0 * ppl / masks.size(0)


# delay emission
class DEHMM(nn.Module):
    def __init__(self, vocab_size, num_state=10):
        super(DEHMM, self).__init__()
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

    @staticmethod
    def bvm_log_product(bm, bv):
        return torch.logsumexp(bm + bv.unsqueeze(-1), dim=-2)

    # log format normalize
    def log_normalize(self, t):
        return self.logsoftmax1(t)

    def forward(self, sentences: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        batch, maxlen = sentences.size()

        # shape [length, batch]
        swapped_sentence = sentences.transpose(0, 1)

        # prob emission (E)
        norm_input = self.logsoftmax0(self.input)
        log_e = F.embedding(swapped_sentence.reshape(-1), norm_input).reshape(maxlen, batch, self.num_state)

        # prob transition (W)
        forward_transition = self.logsoftmax1(self.transition.unsqueeze(0))

        # s_0
        current_mid = self.logsoftmax0(self.begin).expand([batch, self.num_state])

        mid_forwards = [current_mid]
        for i in range(0, maxlen-1):
            # c_i-1
            pre_forward = mid_forwards[i - 1]
            # s_i
            current_forward = self.log_normalize(pre_forward + log_e[i])
            # c_i
            current_mid = self.bvm_log_product(forward_transition, current_forward)
            # current_mid = torch.logsumexp(forward_transition + current_forward.unsqueeze(-1), dim=-1)
            mid_forwards.append(current_mid)
        # shape [max_len, batch, dim]
        hidden_states = torch.stack(mid_forwards)
        # TODO OOM
        pred_prob = self.logsoftmax1(torch.matmul(self.input.unsqueeze(0), torch.exp(hidden_states.view(-1, self.num_state, 1))))
        ppl = torch.logsumexp(pred_prob, dim=-1) * masks.transpose(0, 1)
        return torch.sum(ppl)

    def get_loss(self, sentences: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        ppl = self.forward(sentences, masks)
        return -1.0 * ppl / masks.size(0)


# sigmoid non-linearity
class SNLHMM(nn.Module):
    def __init__(self, vocab_size, num_state=10):
        super(SNLHMM, self).__init__()
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

    @staticmethod
    def bvm_log_product(bm, bv):
        return torch.logsumexp(bm + bv.unsqueeze(-1), dim=-2)

    # log format normalize
    def log_normalize(self, t):
        return self.logsoftmax1(t)

    @staticmethod
    def normalize(t, dim=-1):
        s = torch.sum(t, dim=dim)
        return t / s.unsqueeze(dim)

    def forward(self, sentences: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        batch, maxlen = sentences.size()

        # shape [length, batch]
        swapped_sentence = sentences.transpose(0, 1)
        # prob emission (E)
        norm_input = self.logsoftmax0(self.input)

        emission = F.embedding(swapped_sentence.reshape(-1), norm_input).reshape(maxlen, batch, self.num_state)
        # prob transition (W)
        forward_transition = self.transition.unsqueeze(0)

        # s_0
        current_mid = torch.sigmoid(self.logsoftmax0(self.begin).expand([batch, self.num_state]))
        mid_forwards = [current_mid]
        for i in range(0, maxlen-1):
            # c_i-1
            pre_forward = mid_forwards[i]
            # log format s_i
            current_forward = self.log_normalize(torch.log(self.normalize(pre_forward)) + emission[i])
            # c_i
            current_mid = torch.sigmoid(torch.matmul(forward_transition, torch.exp(current_forward).unsqueeze(-1))).squeeze(-1)
            mid_forwards.append(current_mid)
        # shape [max_len, batch, dim]
        hidden_states = torch.stack(mid_forwards)
        ppl = torch.logsumexp(torch.log(self.normalize(hidden_states)) + emission, dim=-1) * masks.transpose(0, 1)
        return torch.sum(ppl)

    def get_loss(self, sentences: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        ppl = self.forward(sentences, masks)
        return -1.0 * ppl / masks.size(0)