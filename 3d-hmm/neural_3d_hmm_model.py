
import torch
from torch import nn


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
    def __init__(self, xy_size, z_size, num_tokens, token_embeddings=None):
        super(Neural3DHMM, self).__init__()
        num_states = xy_size * xy_size * z_size
        state_embedding_dim = 256
        token_embedding_dim = 256

        # self.state_embeddings = nn.Embedding(num_states, state_embedding_dim)
        # self.token_embeddings = nn.Embedding(num_tokens, token_embedding_dim)

        self.state_embeddings = nn.Parameter(torch.randn(num_states, state_embedding_dim))
        if token_embeddings is None:
            self.token_embeddings = nn.Parameter(torch.randn(num_tokens, token_embedding_dim))
        else:
            assert(len(token_embeddings) == num_tokens)
            token_embedding_dim = token_embeddings.shape[1]
            self.token_embeddings = nn.Parameter(token_embeddings)

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

    def prior_log_p(self):
        priors = self.mlp_start(self.state_embeddings).squeeze(-1)
        return nn.functional.log_softmax(priors, dim=-1)

    def transition_log_p(self):
        h_in = self.mlp_in(self.state_embeddings)
        h_out = self.mlp_out(self.state_embeddings)
        transitions = h_in @ h_out.T
        # TODO restrict to neighborhood
        return nn.functional.log_softmax(transitions, dim=-1)

    def compute_emission_matrix(self):
        h_emit = self.mlp_emit(self.state_embeddings)
        emissions = h_emit @ self.token_embeddings.T
        self.emissions = nn.functional.log_softmax(emissions, dim=1)

    def emission_log_p(self, sentences_tensor):
        emissions = torch.cat((self.emissions, torch.zeros(self.emissions.shape[0], 1)), 1)
        emissions = emissions[:, sentences_tensor].transpose(0, 1)
        return emissions.sum(-1)

    def score(self, stories_tensor, story_length):
        return self.forward_log_p(stories_tensor, story_length)[-1].logsumexp(-1)

    def forward_log_p(self, stories_tensor, story_length):
        assert (stories_tensor.shape[1] == story_length)

        num_stories = stories_tensor.shape[0]

        # p(z0) across all states z0 (Z)
        state_priors = self.prior_log_p()

        # p(z0)*p(x0|z0) across all states z0, all first sentences x0 (N x Z)
        self.compute_emission_matrix()
        emissions = self.emission_log_p(stories_tensor[:, 0])
        scores = [emissions + state_priors]

        transitions = self.transition_log_p()

        for i in range(1, story_length):
            emissions = self.emission_log_p(stories_tensor[:, i])

            # p(zi|zi-1)*p(xi|zi)*p(zi-1)
            intermediate = transitions + scores[-1].view(num_stories, 1, -1)
            scores.append(emissions + intermediate.logsumexp(-1))

        return torch.stack(scores)
