
import torch
from torch import nn
# from torch import tensor


class Gradient3DHMM(nn.Module):
    def __init__(self, xy_size, z_size, num_tokens):
        super(Gradient3DHMM, self).__init__()
        self.xy_size = xy_size
        self.z_size = z_size
        self.num_states = xy_size * xy_size * z_size
        self.num_tokens = num_tokens

        self.transition_matrix_unnormalized = nn.Parameter(torch.randn(self.num_states, 7))
        self.emission_matrix_unnormalized = nn.Parameter(torch.randn(self.num_states, self.num_tokens))
        self.state_priors_unnormalized = nn.Parameter(torch.randn(self.num_states))

    def index2coord(self, state):
        z, state = divmod(state, self.xy_size * self.xy_size)
        y, x = divmod(state, self.xy_size)
        return torch.tensor([x, y, z])

    def prior_log_p(self):
        return nn.functional.log_softmax(self.state_priors_unnormalized, dim=0)

    def transition_log_p(self):
        # transitions = nn.functional.log_softmax(self.transition_matrix_unnormalized, dim=-1)

        # TODO use sparse matrix instead?
        out = torch.zeros(self.num_states, self.num_states).log()

        neighbors = torch.tensor([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, 2]])
        for h in range(self.num_states):
            # print(index2coord(xy_size)(h))
            coord = self.index2coord(h)
            neighbor_coords = neighbors + coord
            neighbor_coords = neighbor_coords[(
                neighbor_coords[:, 0].div(self.xy_size, rounding_mode='floor') == 0
            ).logical_and(
                neighbor_coords[:, 1].div(self.xy_size, rounding_mode='floor') == 0
            ).logical_and(
                neighbor_coords[:, 2].div(self.z_size, rounding_mode='floor') == 0
            )]

            transitions = nn.functional.log_softmax(self.transition_matrix_unnormalized[h, :len(neighbor_coords)], dim=-1)

            for idx, neighbor in enumerate(neighbor_coords):
                x, y, z = neighbor
                i = x + self.xy_size * (y + self.xy_size * z)
                # print("\t", neighbor, "=>", i)
                out[h, i] = transitions[idx]

        return out

    def emission_log_p(self, sentences_tensor):
        # returns matrix with shape num_batches x num_states

        emission_matrix = nn.functional.log_softmax(self.emission_matrix_unnormalized, dim=1)
        emission_matrix = torch.cat((emission_matrix, torch.zeros(emission_matrix.shape[0], 1)), 1)
        # TODO sentence_tensor = sentence_tensor[sentence_tensor > -1]
        emissions = emission_matrix[:, sentences_tensor].transpose(0, 1)
        return emissions.sum(-1)  # TODO normalize??

    def score(self, stories_tensor, story_length):
        return self.forward_log_p(stories_tensor, story_length)[-1].logsumexp(-1)

    def forward_log_p(self, stories_tensor, story_length):
        assert(stories_tensor.shape[1] == story_length)

        num_stories = stories_tensor.shape[0]

        # p(z0) across all states z0 (Z)
        state_priors = self.prior_log_p()
        # print("PRIOR", state_priors.exp())

        # p(z0)*p(x0|z0) across all states z0, all first sentences x0 (N x Z)
        emissions = self.emission_log_p(stories_tensor[:, 0])
        # print("EMITS INITIAL", emissions.exp())
        scores = [emissions + state_priors]
        # print("SCORE INITIAL", scores.exp())
        # print("SCORE INITIAL", scores)

        transitions = self.transition_log_p()
        # print("TRANS", transitions.exp())  # shape is num_states x num_states

        for i in range(1, story_length):
            emissions = self.emission_log_p(stories_tensor[:, i])
            # print(i, "EMITS", emissions.exp())  # shape is num_stories x num_states

            # p(zi|zi-1)*p(xi|zi)*p(zi-1)
            intermediate = transitions + scores[-1].view(num_stories, 1, -1)
            # print(i, "INTER", intermediate.exp())
            scores.append(emissions + intermediate.logsumexp(-1))
            # print(i, "SCORES", scores[-1].exp())
            # print(i, "SCORES", scores[-1])

        return torch.stack(scores)

        # make the log sum
        # sums_log = scores.logsumexp(dim=2)
        # calculate log probabilities
        # TODO log_probabilities = sums_log.gather(1, length.view(-1, 1) - 1)
        # return log_probabilities
