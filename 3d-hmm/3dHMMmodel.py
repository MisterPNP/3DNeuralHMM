from torch import nn


class ResidualLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
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

    def score(self, x):
        pass
        # TODO use parameters from compute_parameters
        #      to score x with the forward algorithm
