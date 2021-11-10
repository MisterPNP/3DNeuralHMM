
import torch


class Neural3DHMM(torch.nn.Module):
    def __init__(self):
        self.start_emb = StateEmbedding(
            self.C,
            config.hidden_dim,
            num_embeddings1=config.num_clusters if config.state == "fac" else None,
            num_embeddings2=config.states_per_word if config.state == "fac" else None,
        )
        self.start_mlp = nn.Sequential(
            ResidualLayer(
                in_dim=config.hidden_dim,
                out_dim=config.hidden_dim,
                dropout=config.dropout,
            ),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

        # p(zt | zt-1)
        self.state_emb = StateEmbedding(
            self.C,
            config.hidden_dim,
            num_embeddings1=config.num_clusters if config.state == "fac" else None,
            num_embeddings2=config.states_per_word if config.state == "fac" else None,
        )
        self.trans_mlp = nn.Sequential(
            ResidualLayer(
                in_dim=config.hidden_dim,
                out_dim=config.hidden_dim,
                dropout=config.dropout,
            ),
            nn.Dropout(config.dropout),
        )
        self.next_state_emb = StateEmbedding(
            self.C,
            config.hidden_dim,
            num_embeddings1=config.num_clusters if config.state == "fac" else None,
            num_embeddings2=config.states_per_word if config.state == "fac" else None,
        )

        # p(xt | zt)
        self.preterminal_emb = StateEmbedding(
            self.C,
            config.hidden_dim,
            num_embeddings1=config.num_clusters if config.state == "fac" else None,
            num_embeddings2=config.states_per_word if config.state == "fac" else None,
        )
        self.terminal_mlp = nn.Sequential(
            ResidualLayer(
                in_dim=config.hidden_dim,
                out_dim=config.hidden_dim,
                dropout=config.dropout,
            ),
            nn.Dropout(config.dropout),
        )
        self.terminal_proj = nn.Linear(config.hidden_dim, len(V))
