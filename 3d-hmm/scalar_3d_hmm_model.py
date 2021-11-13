
import torch
from torch import nn
# from torch import tensor


def index2coord(state, xy_size):
    z, state = divmod(state, xy_size * xy_size)
    y, x = divmod(state, xy_size)
    return torch.tensor([x, y, z])


class TransitionModel(nn.Module):
    def __init__(self, states):
        super(TransitionModel, self).__init__()
        self.states = states
        self.transition_matrix_unnormalized = nn.Parameter(torch.randn(states, 7))

    def log_p(self, state_next, state_prev):
        x, y, z = index2coord(state_next, xy_size) - index2coord(state_prev, xy_size)
        neighbors = [(0,0,0), (1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,2)]
        if (x, y, z) in neighbors:
            # TODO is this correct, using `state_prev`?
            transitions = nn.functional.log_softmax(self.transition_matrix_unnormalized[state_prev], dim=-1)
            return transitions[neighbors.index((x,y,z))]



class EmissionModel(nn.Module):
    def __init__(self, num_states, num_tokens, num_observations):
        super(EmissionModel, self).__init__()
        self.states = num_states
        self.observations = num_observations

        self.emission_matrix_unnormalized = nn.Parameter(torch.randn(num_states, num_tokens))

    def log_p(self, sentences_tensor):
        emission_matrix = nn.functional.softmax(self.emission_matrix_unnormalized, dim=1)
        # sentence_tensor = sentence_tensor[sentence_tensor > -1]
        emissions = emission_matrix[:, sentences_tensor].transpose(0, 1)
        return emissions.log().sum(-1)  # TODO normalize??


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
    def forward(self, stories_tensor, length, batch_size):
        story_length = stories_tensor.shape[1]

        # p(z0) across all states z0 (Z)
        state_priors = nn.functional.log_softmax(self.state_prior_unnormalized, dim=0)
        # p(z0)*p(x0|z0) across all states z0, all first sentences x0 (N x Z)
        scores = self.emission_model.log_p(stories_tensor[:, 0]) + state_priors

        transitions = self.transition_model.log_p()
        for i in range(1, story_length):
            scores = self.emission_model.log_p(stories_tensor[:, i]) + self.transition_model()

        for i in range(1, story_length):
            for state in range(self.num_states):
                for prev in range(self.num_states):
                    scores_next[:, state] +=\
                        self.transition_model.log_p(state, prev)\
                        * self.emission_model.log_p(stories_tensor[:, i])[:,state]\
                        * scores[:, prev]

        # make the log sum
        sums_log = scores.logsumexp(dim=2)
        # calculate log probabilities
        log_probabilities = torch.gather(sums_log, 1, length.view(-1, 1) - 1)

        return log_probabilities

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
