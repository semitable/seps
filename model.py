import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical
from gym.spaces import flatdim


class MultiCategorical:
    def __init__(self, categoricals):
        self.categoricals = categoricals

    def __getitem__(self, key):
        return self.categoricals[key]

    def sample(self):
        return [c.sample().unsqueeze(-1) for c in self.categoricals]

    def log_probs(self, actions):

        return [
            c.log_prob(a.squeeze(-1)).unsqueeze(-1)
            for c, a in zip(self.categoricals, actions)
        ]

    def mode(self):
        return [c.mode for c in self.categoricals]

    def entropy(self):
        return [c.entropy() for c in self.categoricals]


class MultiAgentFCNetwork(nn.Module):
    def _init_layer(self, m):
        nn.init.orthogonal_(m.weight.data, gain=np.sqrt(2))
        nn.init.constant_(m.bias.data, 0)
        return m

    def _make_fc(self, dims, activation=nn.ReLU, final_activation=None):
        mods = []

        input_size = dims[0]
        h_sizes = dims[1:]

        mods = [nn.Linear(input_size, h_sizes[0])]
        for i in range(len(h_sizes) - 1):
            mods.append(activation())
            mods.append(self._init_layer(nn.Linear(h_sizes[i], h_sizes[i + 1])))

        if final_activation:
            mods.append(final_activation())

        return nn.Sequential(*mods)

    def __init__(self, input_sizes, idims):
        super().__init__()

        self.laac_size = len(input_sizes)
        self.independent = nn.ModuleList()

        for size in input_sizes:
            dims = [size] + idims
            self.independent.append(self._make_fc(dims))

    def forward(self, inputs, laac_indices):
        # print(inputs[0].shape)
        # assert inputs[0].dim() == 2
        # out2 = self.forward2(inputs, laac_indices)
        inputs = torch.stack(inputs)
        out = torch.stack([net(inputs) for net in self.independent])
        if inputs[0].dim() == 3:
            laac_indices = laac_indices.T.unsqueeze(0).unsqueeze(-1).unsqueeze(2)
            laac_indices = laac_indices.expand(1, *out.shape[1:])
        else:
            laac_indices = laac_indices.T.unsqueeze(0).unsqueeze(-1).expand(1, *out.shape[1:])

        out = out.gather(0, laac_indices).split(1, dim=1)

        out = [x.squeeze(0).squeeze(0) for x in out]
        # print(out[0].shape)

        # out_test = [self.independent[0](x) for x in inputs]

        # inputs = torch.stack(inputs)
        # shape = inputs.shape
        # inputs = inputs.reshape(-1, shape[-1])
        # out = torch.stack([net(inputs) for net in self.independent])
        # laac_indices = laac_indices.T.reshape(-1, 1).expand_as(out[0]).unsqueeze(0)
        # out = out.gather(0, laac_indices)
        # out_shape = *shape[:-1], out.shape[-1]
        # out = out.reshape(out_shape)

        # print(out.shape)
        # out = out.split(1, dim=0)
        # out = [x.squeeze(0) for x in out]
        # print(out[0].shape)
        return out



class Policy(nn.Module):
    def __init__(self, obs_space, action_space, architecture, laac_size, state_size):
        super(Policy, self).__init__()

        self.n_agents = len(obs_space)
        self.laac_size = laac_size

        obs_space = obs_space[:laac_size]
        action_space = action_space[:laac_size]

        obs_shape = [flatdim(o) for o in obs_space]
        action_shape = [flatdim(a) for a in action_space]

        self.actor = MultiAgentFCNetwork(
            obs_shape, architecture["actor"] + [action_shape[0]]
        )
        for layers in self.actor.independent:
            nn.init.orthogonal_(layers[-1].weight.data, gain=0.01)

        if state_size:
            state_size = len(obs_space) * [state_size]
        else:
            state_size = obs_shape

        self.critic = MultiAgentFCNetwork(
            state_size,
            architecture["critic"] + [1],
        )

        num_outputs = [asp.n for asp in action_space]

        self.laac_params = nn.Parameter(torch.ones(self.n_agents-1, laac_size))
        print(self)

    def sample_laac(self, batch_size):
        sample = Categorical(logits=self.laac_params).sample([batch_size])
        self.laac_sample = torch.cat((torch.zeros(batch_size,1).int(), sample), dim=1)
        # print(self.laac_sample)
        # self.laac_sample = torch.zeros_like(self.laac_sample)

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def get_dist(self, actor_features, action_mask):
        if action_mask:
            action_mask = [-9999999 * (1 - a) for a in action_mask]
        else:
            action_mask = len(actor_features) * [0]

        dist = MultiCategorical(
            [Categorical(logits=x + s) for x, s in zip(actor_features, action_mask)]
        )
        return dist

    def act(self, inputs, action_mask=None):
        actor_features = self.actor(inputs, self.laac_sample)
        dist = self.get_dist(actor_features, action_mask)
        action = dist.sample()
        return action

    def get_value(self, inputs):
        return torch.cat(self.critic(inputs, self.laac_sample), dim=-1)

    def evaluate_actions(self, inputs, action, action_mask=None, state=None):
        if not state:
            state = inputs

        value = self.get_value(state)
        actor_features = self.actor(inputs, self.laac_sample)
        dist = self.get_dist(actor_features, action_mask)
        action_log_probs = torch.cat(dist.log_probs(action), dim=-1)
        dist_entropy = dist.entropy()
        dist_entropy = sum([d.mean() for d in dist_entropy])

        return (
            value,
            action_log_probs,
            dist_entropy,
        )


# define a simple linear VAE
class LinearVAE(nn.Module):
    def __init__(self, features, input_size, extra_decoder_input, reconstruct_size):
        super(LinearVAE, self).__init__()
        HIDDEN=64
        self.features = features
        # encoder
        self.gru = nn.GRU(input_size=input_size, hidden_size=HIDDEN, batch_first=True) # not used for now
        
        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=HIDDEN),
            nn.ReLU(),
            nn.Linear(in_features=HIDDEN, out_features=2*features)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(in_features=features + extra_decoder_input, out_features=HIDDEN),
            nn.ReLU(),
            nn.Linear(in_features=HIDDEN, out_features=HIDDEN),
            nn.ReLU(),
            nn.Linear(in_features=HIDDEN, out_features=reconstruct_size),
        )
 
    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample

    def encode(self, x):
#         x, _ = self.gru(x)
        x = self.encoder(x)
        mu = x[:, :self.features]
        log_var = x[:, self.features:]
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        return mu
        
    def forward(self, x, xp):
        # encoding
#         x, _ = self.gru(x)
        x = self.encoder(x)
        
        mu = x[: , :self.features]
        log_var = x[:, self.features:]
        
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        dec_input = torch.cat([z, xp], axis=-1)      
        # decoding
        reconstruction = self.decoder(dec_input)
        return reconstruction, mu, log_var