
import torch
from torch import nn


class TransitionModel(torch.nn.Module):
    def __init__(self, states):
        super(TransitionModel, self).__init__()
        self.states = states
        self.transition_matrix_unnormalized = torch.nn.Parameter(torch.randn(states, states))


class EmissionModel(torch.nn.Module):
    def __init__(self, states, observations):
        super(EmissionModel, self).__init__()
        self.states = states
        self.observations = observations
        self.emission_matrix_unnormalized = torch.nn.Parameter(torch.randn(states, observations))


class ResidualLayer(nn.Module):
    def __init__(self, in_dim, out_dim,
                 # dropout = 0.,
                 ):
        super(ResidualLayer, self).__init__()
        self.lin1 = nn.Linear(in_dim, out_dim)
        self.lin2 = nn.Linear(out_dim, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)

        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x1 = self.lin1(x).relu()
        # x1 = self.dropout(x1)
        x2 = self.lin2(x1).relu()
        # x2 = self.dropout(x2)
        return self.layer_norm(x2 + x1)


class Neural3DHMM(nn.Module):
    def __init__(self, xy_size, z_size, num_tokens):
        num_states = xy_size * xy_size * z_size
        state_embedding_dim = 256
        token_embedding_dim = 256

        self.states = nn.Embedding(num_states, state_embedding_dim)
        self.tokens = nn.Embedding(num_tokens, token_embedding_dim)

        # A matrix in terms of number of states (out_dim)
        self.transition_model = TransitionModel(self.xy_size)

        # emission model
        self.emission_model = EmissionModel(self.xy_size, self.z_size)

        self.state_priors_unnormalized = torch.nn.Parameter(torch.randn(self.states))

        # p(z0)
        intermediate_dim = 256
        self.mlp_start = nn.Sequential(
            ResidualLayer(
                in_dim=state_embedding_dim,
                out_dim=intermediate_dim,
            ),
            nn.Linear(intermediate_dim, 1),
        )

        # p(zt | zt-1)
        intermediate_dim = 256
        self.mlp_in = ResidualLayer(
            in_dim=state_embedding_dim,
            out_dim=intermediate_dim,
        )
        self.mlp_out = ResidualLayer(
            in_dim=state_embedding_dim,
            out_dim=intermediate_dim,
        )

        # p(xt | zt)
        self.mlp_emit = ResidualLayer(
            in_dim=state_embedding_dim,
            out_dim=token_embedding_dim,
        )

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
    def forward(self, input_tensor, length, batch_size):
        length_max = input_tensor.shape[1]

        state_priors = torch.nn.functional.log_softmax(self.state_prior_unnormalized, dim=0)
        alpha = torch.zeros(batch_size, length_max, self.N)

        alpha[:, 0, :] = self.emission_model(input_tensor[:, 0]) + state_priors

        for i in range(1, length_max):
            alpha[:, i, :] = self.emission_model(input_tensor[:, t]) + self.transition_model(alpha[:, i - 1, :])

        # make the log sum
        sums_log = alpha.logsumexp(dim=2)
        # calculate log probabilities
        log_probabilities = torch.gather(sums_log, 1, length.view(-1, 1) - 1)

        return log_probabilities

    # emissions forward algorithm
    def emissions_forward(self, x):
        log_emission = torch.nn.functional.log_softmax(self.emission_matrix_unnormalized, dim=1)
        return log_emission[:, x].transpose(0, 1)

    # transitions forward algorithm
    def transitions_forward(self, alpha):
        log_transition = torch.nn.functional.log_softmax(self.transition_matrix_unnormalized, dim=0)
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
