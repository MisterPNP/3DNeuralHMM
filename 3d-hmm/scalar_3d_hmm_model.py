
import torch
from torch import nn
# from torch import tensor


class Scalar3DHMM(nn.Module):
    def __init__(self, xy_size, z_size, num_tokens):
        super(Scalar3DHMM, self).__init__()
        self.xy_size = xy_size
        self.z_size = z_size
        self.num_states = xy_size * xy_size * z_size
        self.num_tokens = num_tokens

        self.transition_matrix_unnormalized = nn.Parameter(torch.randn(self.num_states, 7))
        self.emission_matrix_unnormalized = nn.Parameter(torch.randn(self.num_states, self.num_tokens))
        print(self.emission_matrix_unnormalized.grad)
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

    def backward_log_p(self, stories_tensor, story_length):
        assert(stories_tensor.shape[1] == story_length)
        num_stories = stories_tensor.shape[0]

        scores = [torch.ones(num_stories, self.num_states)]  # N x Z
        transitions = self.transition_log_p()  # Z x Z
        for i in range(story_length - 1, 0, -1):
            emissions = self.emission_log_p(stories_tensor[:, i])  # N x Z
            intermediate = transitions + scores[-1].view(num_stories, -1, 1) \
                           + emissions.view(num_stories, 1, -1)  # N x Z x Z
            scores.append(intermediate.logsumexp(-1))  # N x Z

        scores.reverse()
        return torch.stack(scores)

    def baum_welch_updates(self, stories_tensor):
        num_stories = stories_tensor.shape[0]
        story_length = stories_tensor.shape[1]

        t = self.transition_log_p().exp()  # Z x Z
        e = self.emission_log_p(
                stories_tensor.view(num_stories * story_length, -1)
            ).exp().view(num_stories, story_length, -1)  # N x L x Z
        a = self.forward_log_p(stories_tensor, story_length).exp().transpose(0, 1)  # N x L x Z
        b = self.backward_log_p(stories_tensor, story_length).exp().transpose(0, 1)  # N x L x Z

        print(t, e, a, b)

        p_state_given_story = ((a * b).T / (a * b).sum(-1).T).T  # N x L x Z
        p_pair_given_story = ((a[:, :story_length - 1].view(num_stories, story_length - 1, -1, 1) * t
                              * (e * b)[:, 1:].view(num_stories, story_length - 1, 1, -1)).T
                              / a[:, -1].sum(-1)).T  # N x L-1 x Z x Z

        print(p_state_given_story, p_pair_given_story)  # TODO why are these NaN?

        priors = p_state_given_story[:, 0].mean(0)  # Z x 1
        emissions = torch.zeros(self.num_states, self.num_tokens)  # Z x K
        # TODO vectorize?
        for story in stories_tensor:
            for i, sentence in enumerate(story):
                sentence_length = (sentence > -1).sum()
                for token in sentence[sentence > -1]:
                    emissions[:, token] += p_state_given_story[:, i].squeeze(1).sum(0) / sentence_length
        emissions = (emissions.T / p_state_given_story.sum((0, 1))).T
        transitions = p_pair_given_story.sum((0, 1)) / p_state_given_story.sum((0, 1))  # Z x Z

        return priors, transitions, emissions
