import math
import time
from functools import partial

import numpy as np
import torch
from torch import nn


class Scalar3DHMM(nn.Module):
    def __init__(self, xy_size, z_size, num_tokens, win_size=2):
        super(Scalar3DHMM, self).__init__()
        self.xy_size = xy_size
        self.z_size = z_size
        self.num_states = xy_size * xy_size * z_size
        self.num_tokens = num_tokens
        self.win_size = win_size

        self.transitions = nn.Parameter(self.initialize_transitions())
        self.emissions = nn.Parameter(nn.functional.log_softmax(torch.randn(self.num_states, self.num_tokens), dim=-1))
        self.emission_matrix_precomputed = None
        self.priors = nn.Parameter(nn.functional.log_softmax(torch.randn(self.num_states), dim=-1))

    def set_paramters(self, priors, transitions, emissions):
        self.priors = priors
        self.transitions = transitions
        self.emissions = emissions
        self.emission_matrix_precomputed = None

    def index2coord(self, state):
        z, state = divmod(state, self.xy_size * self.xy_size)
        y, x = divmod(state, self.xy_size)
        return torch.tensor([x, y, z])

    def initialize_transitions(self):
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

            transitions = nn.functional.log_softmax(torch.randn(len(neighbor_coords)), dim=-1)

            for idx, neighbor in enumerate(neighbor_coords):
                x, y, z = neighbor
                i = x + self.xy_size * (y + self.xy_size * z)
                # print("\t", neighbor, "=>", i)
                out[h, i] = transitions[idx]

        return out

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
        return self.priors

    def transition_log_p(self):
        return self.transitions

    def emission_log_p(self, sentences_tensor):
        # compute window-smoothed emission matrix
        if self.emission_matrix_precomputed is not None:
            emission_matrix = self.emission_matrix_precomputed
        else:
            emission_matrix = self.emissions.view(self.z_size, self.xy_size, self.xy_size, self.num_tokens)
            emission_matrix = self.compute_windowed_log_mean(emission_matrix, dims=(-2, -3))
            emission_matrix = emission_matrix.contiguous().view(self.num_states, self.num_tokens)
            self.emission_matrix_precomputed = emission_matrix

        # take product of emissions from sentence indices
        emission_matrix = nn.functional.pad(emission_matrix, (0, 1))  # Z x K+1
        emissions = emission_matrix[:, sentences_tensor].transpose(0, 1)  # N x Z x T
        emissions_sum = emissions.sum(-1)  # N x Z

        # TODO normalize by sentence length ??
        # emissions_sum = (emissions_sum.T / (sentences_tensor > -1).sum(-1)).T

        return emissions_sum

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

        scores = [torch.ones(num_stories, self.num_states, device=self.transitions.device)]  # N x Z
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
                stories_tensor.contiguous().view(num_stories * story_length, -1)
            ).view(num_stories, story_length, -1)  # N x L x Z
        a = self.forward_log_p(stories_tensor, story_length).transpose(0, 1)  # N x L x Z
        print("finished forward pass")
        b = self.backward_log_p(stories_tensor, story_length).transpose(0, 1)  # N x L x Z
        print("finished backward pass")

        # print("TRANS", t.exp())
        # print("EMITS", e.exp())
        # print("FORWD", a.exp())
        # print("BAKWD", b.exp())

        def normalize(x, dim=-1):
            return (x.T - x.logsumexp(dim).T).T

        p_states = normalize(a + b)  # N x L x Z
        p_pairs = normalize(
            a[:, :story_length - 1].view(num_stories, story_length - 1, -1, 1) + t
            + (e + b)[:, 1:].view(num_stories, story_length - 1, 1, -1), dim=(-1, -2))  # N x L-1 x Z x Z

        # print("p(STEP,STATE)", p_states.exp())
        # print(p_states.logsumexp(-1).exp())  # this should be all ones
        # print("p(STEP,PAIR) ", p_pairs.exp())
        # print(p_pairs.logsumexp((-1, -2)).exp())  # this should be all ones

        log_num_stories = torch.tensor(num_stories, device=stories_tensor.device).log()
        log_story_length = torch.tensor(story_length, device=stories_tensor.device).log()

        priors = p_states[:, 0].logsumexp(0) - log_num_stories  # Z x 1
        transitions = p_pairs.logsumexp((0, 1)) - log_num_stories - log_story_length  # Z x Z

        # print("PRIOR", priors.exp())
        # print(priors.logsumexp(-1).exp())  # should be one
        # print("TRANS", transitions.exp())
        # print(transitions.logsumexp(-1).exp())  # should be ones

        # sentence_lengths = (stories_tensor > -1).sum(-1)  # N x L
        # sentence_lengths[sentence_lengths == 0] = 1
        # token_counts = torch.stack(tuple(map(
        #     partial(torch.bincount, minlength=self.num_tokens + 1),
        #     (stories_tensor + 1).flatten(end_dim=-2).unbind()
        # )))[:, 1:].view(num_stories, story_length, self.num_tokens)  # N x L x K
        # token_hists = (token_counts.float().T / sentence_lengths.T).T.sum(0)  # L x K
        # p_state_mean = p_states.logsumexp(0) - log_num_stories  # L x Z
        # emissions = ((p_state_mean.T.view(-1, 1, story_length)
        #               + token_hists.T.view(1, -1, story_length).log()).logsumexp(-1).T
        #              - p_states.logsumexp((0, 1)).T).T  # Z x K

        # π'_i,z ∝ π_i,z * Σ_d c_d,z Σ_k|i∈W_k P(i_d=k) / h_k,z
        emission_matrix = self.emission_matrix_precomputed  # Z x K
        scale = []
        batch_size = 20
        for i in range(math.ceil(num_stories / batch_size)):
            print("{0:0>5.2f}%".format(100 * i * batch_size / num_stories), end="\r")
            p_states_batch = p_states[i * batch_size:(i+1) * batch_size].contiguous()
            p_states_over_emissions = (p_states_batch.view(-1, story_length, 1, self.num_states)
                                       - emission_matrix.view(1, 1, self.num_tokens, self.num_states))  # n x L x K x Z
            p_states_over_emissions = p_states_over_emissions.contiguous().view(-1, self.xy_size, self.xy_size)
            p_states_over_emissions = self.compute_windowed_log_mean(p_states_over_emissions, dims=(-1, -2))
            p_states_over_emissions = p_states_over_emissions.contiguous().view(
                                          -1, story_length, self.num_tokens, self.num_states)

            batch_tensor = stories_tensor[i * batch_size:(i+1) * batch_size]
            token_counts = torch.stack(tuple(map(
                partial(torch.bincount, minlength=self.num_tokens + 1),
                (batch_tensor + 1).flatten(end_dim=-2).unbind()
            )))[:, 1:].view(-1, story_length, self.num_tokens)  # n x L x K

            p_scaled = (p_states_over_emissions.T + token_counts.T).T  # n x L x K x Z
            scale.append(p_scaled.logsumexp((0, 1)).T)  # Z x K
            if len(scale) > 1000:
                scale = [torch.stack(scale).logsumexp(0)]
        scales = torch.stack(scale).logsumexp(0)
        emissions = normalize(emission_matrix + scales)  # Z x K

        # print("EMITS", emissions.exp())
        # print(emissions.logsumexp(-1).exp())  # should be ones

        return nn.Parameter(priors), nn.Parameter(transitions), nn.Parameter(emissions)
