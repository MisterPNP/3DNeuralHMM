
import torch
from torch import nn
# from torch import tensor


# def index2coord(xy_size):
#     def f(state):
#         z, state = divmod(state, xy_size * xy_size)
#         y, x = divmod(state, xy_size)
#         return torch.tensor([x, y, z])
#     return f

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

    def transition_log_p(self):
        transitions = nn.functional.log_softmax(self.transition_matrix_unnormalized, dim=-1)

        # TODO use sparse matrix instead?
        out = torch.zeros(self.num_states, self.num_states).log()

        neighbors = [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, 2)]
        for h in range(self.num_states):
            # print(index2coord(xy_size)(h))

            # TODO distribution is not always over seven states!!!
            for idx, neighbor in enumerate(neighbors):
                # i = coord2index(index2coord(h) - neighbor)
                x, y, z = neighbor
                i = (h - (x + self.xy_size * (y + self.xy_size * z))) % self.num_states  # TODO don't wrap, but check bounds
                # print("\t", neighbor, "=>", i)
                out[h, i] = transitions[h, idx]

        return out

    def emission_log_p(self, sentences_tensor):
        # returns matrix with shape num_batches x num_states

        # sentences_tensor = sentences_tensor.clamp(max=11)  # TODO testing

        emission_matrix = nn.functional.log_softmax(self.emission_matrix_unnormalized, dim=1)
        # TODO sentence_tensor = sentence_tensor[sentence_tensor > -1]
        emissions = emission_matrix[:, sentences_tensor].transpose(0, 1)
        return emissions.sum(-1)  # TODO normalize??

    def score(self, stories_tensor, story_length):
        return self.forward_log_p(stories_tensor, story_length)[-1].logsumexp(-1)

    def forward_log_p(self, stories_tensor, story_length):
        assert(stories_tensor.shape[1] == story_length)

        num_stories = stories_tensor.shape[0]

        # p(z0) across all states z0 (Z)
        state_priors = nn.functional.log_softmax(self.state_priors_unnormalized, dim=0)
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

        scores = torch.ones(num_stories, self.num_states)  # N x Z
        transitions = self.transition_log_p()  # Z x Z
        for i in range(story_length - 1, 0, -1):
            emissions = self.emission_log_p(stories_tensor[:, i])  # N x Z
            intermediate = transitions + scores[-1].view(num_stories, -1, 1) \
                           + emissions.view(num_stories, 1, -1)  # N x Z x Z
            scores.append(intermediate.logsumexp(-1))  # N x Z

        scores.reverse()
        return torch.stack(scores)

    def baum_welch_updates(self, story):
        story_length = story.shape[0]
        t = self.transition_log_p()  # Z x Z
        e = self.emission_log_p(story)  # L x Z
        a = self.forward_log_p(torch.tensor([story]), story_length).view(story_length, -1)  # L x Z
        b = self.backward_log_p(torch.tensor([story]), story_length).view(story_length, -1)  # L x Z
        p_state_given_story = a * b / (a * b).sum(-1)  # L x Z
        p_pair_given_story = a[:story_length - 1].view(story_length, -1, 1) \
                             * t * (e * b)[1:].view(story_length, 1, -1) / a[-1].sum()  # L x Z x Z
        priors = p_state_given_story[0]  # Z x 1
                                     # TODO only columns where the kth token is emitted at the zth step...
        emissions = p_state_given_story[:].sum(0) / p_state_given_story.sum(0)  # Z x K
        transitions = p_pair_given_story.sum(0) / p_state_given_story.sum(0)  # Z x Z
        return priors, transitions, emissions
