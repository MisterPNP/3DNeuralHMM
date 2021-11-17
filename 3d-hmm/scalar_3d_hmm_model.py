
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

        self.transitions = self.initialize_transitions()
        self.emissions = nn.functional.softmax(torch.randn(self.num_states, self.num_tokens), dim=-1)
        self.priors = nn.functional.softmax(torch.randn(self.num_states), dim=-1)

    def index2coord(self, state):
        z, state = divmod(state, self.xy_size * self.xy_size)
        y, x = divmod(state, self.xy_size)
        return torch.tensor([x, y, z])

    def initialize_transitions(self):
        # TODO use sparse matrix instead?
        out = torch.zeros(self.num_states, self.num_states)

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

            transitions = nn.functional.softmax(torch.randn(len(neighbor_coords)), dim=-1)

            for idx, neighbor in enumerate(neighbor_coords):
                x, y, z = neighbor
                i = x + self.xy_size * (y + self.xy_size * z)
                # print("\t", neighbor, "=>", i)
                out[h, i] = transitions[idx]

        return out

    def prior_log_p(self):
        return self.priors.log()

    def transition_log_p(self):
        return self.transitions.log()

    def emission_log_p(self, sentences_tensor):
        # returns matrix with shape num_batches x num_states

        emission_matrix = self.emissions.log()
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
            intermediate = (transitions + scores[-1].view(num_stories, -1, 1)
                            + emissions.view(num_stories, 1, -1))  # N x Z x Z
            scores.append(intermediate.logsumexp(-1))  # N x Z

        scores.reverse()
        return torch.stack(scores)

    def baum_welch_updates(self, stories_tensor):
        num_stories = stories_tensor.shape[0]
        story_length = stories_tensor.shape[1]

        t = self.transition_log_p()  # Z x Z
        e = self.emission_log_p(
                stories_tensor.view(num_stories * story_length, -1)
            ).view(num_stories, story_length, -1)  # N x L x Z
        a = self.forward_log_p(stories_tensor, story_length).transpose(0, 1)  # N x L x Z
        b = self.backward_log_p(stories_tensor, story_length).transpose(0, 1)  # N x L x Z

        # print("TRANS", t.exp())
        # print("EMITS", e.exp())
        # print("FORWD", a.exp())
        # print("BAKWD", b.exp())

        p_state_given_story = ((a + b).T - (a + b).logsumexp(-1).T).T  # N x L x Z
        p_pair_given_story = ((a[:, :story_length - 1].view(num_stories, story_length - 1, -1, 1) + t
                               + (e + b)[:, 1:].view(num_stories, story_length - 1, 1, -1)).T
                              - a[:, -1].logsumexp(-1).T).T  # N x L-1 x Z x Z

        # TODO this normalization shouldn't be necessary
        p_pair_given_story = (p_pair_given_story.T - p_pair_given_story.logsumexp((-1, -2)).T).T

        # print("p(STEP,STATE)", p_state_given_story.exp())
        # print(p_state_given_story.logsumexp(-1).exp()) # this should be all ones
        # print("p(STEP,PAIR) ", p_pair_given_story.exp())
        # print(p_pair_given_story.logsumexp((-1, -2)).exp())  # this should be all ones

        priors = p_state_given_story[:, 0].exp().mean(0)  # Z x 1
        emissions = torch.zeros(self.num_states, self.num_tokens)  # Z x K
        # TODO vectorize?
        for story in stories_tensor:
            for i, sentence in enumerate(story):
                sentence_length = (sentence > -1).sum()
                for token in sentence[sentence > -1]:
                    emissions[:, token] += p_state_given_story[:, i].squeeze(1).logsumexp(0).exp() / sentence_length
        emissions = (emissions.T / num_stories / p_state_given_story.logsumexp((0, 1)).exp()).T
        transitions = (p_pair_given_story.logsumexp((0, 1)) - p_state_given_story.logsumexp((0, 1))).exp()  # Z x Z

        # TODO this normalization shouldn't be necessary
        transitions = (transitions.T / transitions.sum(-1).T).T

        # print("PRIOR", priors)
        # print(priors.sum(-1))  # should be one
        # print("EMITS", emissions)
        # print(emissions.sum(-1))  # should be ones
        # print("TRANS", transitions)
        # print(transitions.sum(-1))  # should be ones

        return priors, transitions, emissions
