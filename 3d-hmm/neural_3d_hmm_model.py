
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
    def __init__(self, xy_size, z_size, num_tokens, win_size=2, token_embeddings=None):
        super(Neural3DHMM, self).__init__()
        self.xy_size = xy_size
        self.z_size = z_size
        self.num_states = xy_size * xy_size * z_size
        self.num_tokens = num_tokens
        self.win_size = win_size
        state_embedding_dim = 256
        token_embedding_dim = 100

        # self.state_embeddings = nn.Embedding(self.num_states, state_embedding_dim)
        # self.token_embeddings = nn.Embedding(num_tokens, token_embedding_dim)

        self.state_embeddings = nn.Parameter(torch.randn(self.num_states, state_embedding_dim))
        if token_embeddings is None:
            self.token_embeddings = nn.Parameter(torch.randn(num_tokens, token_embedding_dim))
        else:
            assert(len(token_embeddings) == num_tokens)
            token_embedding_dim = token_embeddings.shape[1]
            # self.token_embeddings = nn.Parameter(token_embeddings)
            self.token_embeddings = token_embeddings

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

    def index2coord(self, state):
        z, state = divmod(state, self.xy_size * self.xy_size)
        y, x = divmod(state, self.xy_size)
        return torch.tensor([x, y, z])

    def compute_windowed_log_mean(self, matrix, dims):
        win_length = 1 + 2 * self.win_size
        log_win_length = torch.tensor(win_length, device=matrix.device).log()
        shape = matrix.shape
        n_dims = len(shape)
        for dim in dims:
            assert -n_dims <= dim <= n_dims - 1
            dim = (dim + n_dims) % n_dims
            pad = (n_dims - 1 - dim) * (0, 0) + (win_length - 1, win_length - 1)
            matrix = nn.functional.pad(matrix, pad, mode='replicate')
            matrix = matrix.unfold(dimension=dim, size=win_length, step=1).logsumexp(-1) - log_win_length
            matrix = matrix.view(-1, *matrix.shape[dim:])[:, self.win_size:-self.win_size].view(shape)
        return matrix

    def prior_log_p(self):
        priors = self.mlp_start(self.state_embeddings).squeeze(-1)
        return nn.functional.log_softmax(priors, dim=-1)

    def transition_log_p(self):
        h_in = self.mlp_in(self.state_embeddings)  # Z x h
        h_out = self.mlp_out(self.state_embeddings)  # Z x h

        # # pad h_out with log zeros
        # h_out = h_out.view(self.z_size, self.xy_size, self.xy_size, -1)  # z x xy x xy x h
        # h_out = nn.functional.pad(h_out, (0, 0, 1, 1, 1, 1, 0, 2),
        #                           mode='constant', value=torch.tensor(0).log())  # z+2 x xy+2 x xy+2 x h
        # # index h_out at neighborhoods
        # xy_size =
        # neighbors = torch.arange(self.num_states) + torch.tensor([
        #     0, 1, -1, self.xy_size, -self.xy_size,
        #     (self.xy_size + 2) * self.xy_size, 2 * self.xy_size * self.xy_size])  # Z x 7
        # h_out = h_out[neighbors]  # Z x 7 x h
        #
        # transitions = (h_out @ h_in.unsqueeze(-1)).squeeze(-1)  # Z x 7

        transitions = h_in @ h_out.T

        neighbors = torch.arange(self.num_states).unsqueeze(-1) + torch.tensor([
            0, 1, -1, self.xy_size, -self.xy_size,
            self.xy_size * self.xy_size, 2 * self.xy_size * self.xy_size])  # Z x 7
        mask = (torch.arange(self.num_states).unsqueeze(-1).unsqueeze(-1) != neighbors).all(dim=-1)  # Z x Z
        transitions[mask.T] = torch.tensor(0).log()

        return nn.functional.log_softmax(transitions, dim=-1)

    def compute_emission_matrix(self):
        h_emit = self.mlp_emit(self.state_embeddings)
        emissions = h_emit @ self.token_embeddings.T
        emission_matrix = nn.functional.log_softmax(emissions, dim=1)
        # smoothing?
        emission_matrix = emission_matrix.view(self.z_size, self.xy_size, self.xy_size, self.num_tokens)
        # lvert, rvert = emission_matrix[:, :, :1, :], emission_matrix[:, :, -1:, :]
        # lhorz, rhorz = emission_matrix[:, :1, :, :], emission_matrix[:, -1:, :, :]
        # c = torch.zeros(self.z_size, 1, 1, self.num_tokens)
        # t = torch.cat
        # up = t((t((lvert, emission_matrix, rvert), 2), t((c, rhorz, c), 2), t((c, rhorz, c), 2)), 1)
        # dn = t((t((c, lhorz, c), 2), t((c, lhorz, c), 2), t((lvert, emission_matrix, rvert), 2)), 1)
        # lf = t((t((lhorz, c, c), 2), t((emission_matrix, rvert, rvert), 2), t((rhorz, c, c), 2)), 1)
        # rt = t((t((c, c, lhorz), 2), t((lvert, lvert, emission_matrix), 2), t((c, c, rhorz), 2)), 1)
        # cr = t((t((c, lhorz, c), 2), t((lvert, emission_matrix, rvert), 2), t((c, rhorz, c), 2)), 1)
        # mean = torch.stack((up, dn, lf, rt, cr)).logsumexp(0) - torch.tensor(5).log()
        # emission_matrix = mean[:, 1:-1, 1:-1].contiguous().view(self.num_states, self.num_tokens)
        emission_matrix = self.compute_windowed_log_mean(emission_matrix, dims=(-2, -3))
        emission_matrix = emission_matrix.contiguous().view(self.num_states, self.num_tokens)
        self.emission_matrix = emission_matrix

    def emission_log_p(self, sentences_tensor):
        emissions = torch.cat((self.emission_matrix, torch.zeros(self.num_states, 1)), 1)
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
