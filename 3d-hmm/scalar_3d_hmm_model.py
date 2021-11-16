
import torch
from torch import nn
# from torch import tensor


def index2coord(xy_size):
    def f(state):
        z, state = divmod(state, xy_size * xy_size)
        y, x = divmod(state, xy_size)
        return torch.tensor([x, y, z])
    return f


class TransitionModel(nn.Module):
    def __init__(self, states):
        super(TransitionModel, self).__init__()
        self.states = states
        self.transition_matrix_unnormalized = nn.Parameter(torch.randn(states, 7))

    def log_p(self, num_states, xy_size):
        transitions = nn.functional.log_softmax(self.transition_matrix_unnormalized, dim=-1)

        out = torch.zeros(num_states, num_states).log()

        neighbors = [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, 2)]
        for h in range(num_states):
            # print(index2coord(xy_size)(h))

            # TODO distribution is not always over seven states!!!
            for idx, neighbor in enumerate(neighbors):
                # i = coord2index(index2coord(h) - neighbor)
                x, y, z = neighbor
                i = (h - (x + xy_size * (y + xy_size * z))) % num_states  # TODO don't wrap, but check bounds
                # print("\t", neighbor, "=>", i)
                out[h, i] = transitions[h, idx]

        return out



class EmissionModel(nn.Module):
    def __init__(self, num_states, num_tokens, num_observations):
        super(EmissionModel, self).__init__()
        self.states = num_states
        self.observations = num_observations

        self.emission_matrix_unnormalized = nn.Parameter(torch.randn(num_states, num_tokens))

    def log_p(self, sentences_tensor):
        # returns matrix with shape num_batches x num_states

        # sentences_tensor = sentences_tensor.clamp(max=11)  # TODO testing

        emission_matrix = nn.functional.log_softmax(self.emission_matrix_unnormalized, dim=1)
        # TODO sentence_tensor = sentence_tensor[sentence_tensor > -1]
        emissions = emission_matrix[:, sentences_tensor].transpose(0, 1)
        return emissions.sum(-1)  # TODO normalize??


class Scalar3DHMM(nn.Module):
    def __init__(self, xy_size, z_size, num_tokens):
        super(Scalar3DHMM, self).__init__()
        self.xy_size = xy_size
        self.z_size = z_size
        self.num_states = xy_size * xy_size * z_size
        self.num_tokens = num_tokens

        # A matrix in terms of number of states (out_dim)
        self.transition_model = TransitionModel(self.num_states)

        self.emission_model = EmissionModel(self.num_states, self.num_tokens, self.z_size)

        self.state_priors_unnormalized = nn.Parameter(torch.randn(self.num_states))

    def compute_parameters(self):
        starts = self.mlp_start(self.states)

        h_in = self.mlp_in(self.states)
        h_out = self.mlp_out(self.states)
        transitions = h_in @ h_out.t()

        h_emit = self.mlp_emit(self.states)
        emissions = h_emit @ self.token_embeddings

        # TODO transform to probabilities (softmax?)

        return starts, transitions, emissions

    # forward algorithm
    def forward(self, stories_tensor, story_length, length):
        assert(stories_tensor.shape[1] == story_length)

        num_batches = stories_tensor.shape[0]

        # p(z0) across all states z0 (Z)
        state_priors = nn.functional.log_softmax(self.state_priors_unnormalized, dim=0)
        # print("PRIOR", state_priors.exp())

        # p(z0)*p(x0|z0) across all states z0, all first sentences x0 (N x Z)
        emissions = self.emission_model.log_p(stories_tensor[:, 0])
        # print("EMITS INITIAL", emissions.exp())
        scores = emissions + state_priors
        # print("SCORE INITIAL", scores.exp())
        # print("SCORE INITIAL", scores)

        # TODO generate sparse matrix?
        transitions = self.transition_model.log_p(self.num_states, self.xy_size)
        # print("TRANS", transitions.exp())  # shape is num_states x num_states

        for i in range(1, story_length):
            # transitions = self.transition_model.log_p(self.num_states, index2coord(self.xy_size))
            # print("\t\t", "TRANS", transitions.shape)  # shape is num_states x num_states
            emissions = self.emission_model.log_p(stories_tensor[:, i])
            # print(i, "EMITS", emissions.exp())  # shape is num_batches x num_states

            # p(zi|zi-1)*p(xi|zi)*p(zi-1)
            intermediate = (emissions.view(num_batches, -1, 1) + scores.view(num_batches, 1, -1))
            # print(i, "INTER", intermediate.exp())
            update = transitions + intermediate  # shape is num_batches x num_states x num_states
            # print(i, "UPDATE", update.exp())

            scores = update.logsumexp(-1)
            # print(i, "SCORES", scores.exp())
            # print(i, "SCORES", scores)

        return scores.logsumexp(-1)

        # make the log sum
        # sums_log = scores.logsumexp(dim=2)
        # calculate log probabilities
        # TODO log_probabilities = sums_log.gather(1, length.view(-1, 1) - 1)
        # return log_probabilities

    # emissions forward algorithm
    def emissions_forward(self, x):
        log_emission = nn.functional.log_softmax(self.emission_matrix_unnormalized, dim=1)
        return log_emission[:, x].transpose(0, 1)

    # transitions forward algorithm
    def transitions_forward(self, alpha):
        log_transition = nn.functional.log_softmax(self.transition_matrix_unnormalized, dim=0)
        return self.log_multiplication(log_transition, alpha.transpose(0, 1)).transpose(0, 1)

    def log_multiplication(self, A, B):
        m = A.shape[0]
        n = A.shape[1]
        p = B.shape[1]

        sum_log_elements = torch.reshape(A, (m, n, 1)) + torch.reshape(B, (1, n, p))

        return torch.logsumexp(sum_log_elements, dim=1)

    def score(self, x):
        pass
        # TODO use parameters from compute_parameters
        #      to score x with the forward algorithm
