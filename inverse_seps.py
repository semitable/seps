import gym
from numpy.core.einsumfunc import _update_other_results
import torch
from torch import nn
from torch.nn.modules.linear import Linear
from model import Policy, LinearVAE
from blazingma.utils.wrappers import RecordEpisodeStatistics
import numpy as np
from cpprb import ReplayBuffer
from ops_utils import rbDataSet
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
import random

class IndexShuffler(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # self.shuffled_indexes = [i for i in range(self.n_agents)]
        # print(self.shuffled_indexes)
        self.shuffled_indexes = [0, 1, 2, 3, 4, 5]
        random.shuffle(self.shuffled_indexes)
        print(self.shuffled_indexes)

    
    def reset(self):
        random.shuffle(self.shuffled_indexes)
        print(self.shuffled_indexes)
        observation = super().reset()
        observation = [observation[i] for i in self.shuffled_indexes]
        return observation

    def step(self, action):
        actions = self.n_agents * [ None ]
        for i in range(self.n_agents):
            actions[self.shuffled_indexes[i]] = action[i]

        observation, reward, done, info = super().step(actions)

        observation = [observation[i] for i in self.shuffled_indexes]
        reward = [reward[i] for i in self.shuffled_indexes]
        done = [done[i] for i in self.shuffled_indexes]
        
        return observation, reward, done, info


def reverse_ops(rb, model, agent_count):
    dataset = rbDataSet(rb.get_all_transitions())
    dataloader = DataLoader(dataset, batch_size=2000, shuffle=True)

    _, (real_agent_index, decoder_in, y) = next(enumerate(dataloader))
    batch_size = y.shape[0]
    result_map = defaultdict(list)

    for i in range(agent_count): # real index
        one_hot_i = torch.nn.functional.one_hot(torch.tensor(i), agent_count).repeat(batch_size, 1).float()
        eq = torch.where((one_hot_i == real_agent_index).all(dim=1))[0]

        for j in range(agent_count): # fake index

            one_hot_j = torch.nn.functional.one_hot(torch.tensor(j), agent_count).repeat(batch_size, 1).float()
            z = model.encode(one_hot_j[eq])
            yn = model.decoder(torch.cat([z, decoder_in[eq]], axis=-1) )
            # yn, _, _ = model(one_hot_j[eq], decoder_in[eq])
            err = ((y[eq] - yn)**2).mean().item()
            result_map[i].append(err)

    for k, v in result_map.items():
        result_map[k] = np.argmin(v)
    return result_map

def main():
    gymkey = "robotic_warehouse:rware-2color-tiny-6ag-v1"
    scale = 3.14
    architecture = {
        "actor": [int(scale*128), int(scale*128)],
        "critic": [int(scale*128), int(scale*128)],
    }
    ops_clusters = 2

    env = RecordEpisodeStatistics(IndexShuffler(gym.make(gymkey)))
    agent_count = len(env.observation_space)
    print(agent_count)

    model = Policy(env.observation_space, env.action_space, architecture, 1, None)
    # model.laac_params = nn.Parameter(torch.ones(3, ops_clusters))
    # model.load_state_dict(torch.load("rl_model.pt"))
    # model.laac_params = nn.Parameter(torch.ones(5, ops_clusters))

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    exit()

    vae = LinearVAE(10, agent_count, 49, 45)
    vae.load_state_dict(torch.load("ops_model.pt"))
    print(vae)

    original_laac = torch.tensor([1, 1, 1, 0, 0, 0]).long()
    model.laac_sample = original_laac

    obs_size = env.observation_space[0].shape
    act_size = env.action_space[0].n

    # replay buffer:
    env_dict = {
        "obs": {"shape": obs_size, "dtype": np.float32},
        "rew": {"shape": 1, "dtype": np.float32},
        "next_obs": {"shape": obs_size, "dtype": np.float32},
        "done": {"shape": 1, "dtype": np.float32},
        "act": {"shape": act_size, "dtype": np.float32},
        "agent": {"shape": agent_count , "dtype": np.float32},
    }
    rb = ReplayBuffer(int(env.n_agents * 500), env_dict)

    accuracies = []
    accuracy = []
    obs = [torch.from_numpy(o) for o in env.reset()]
    for _ in range(100000):
        act = model.act(obs)
        nobs, rew, done, info = env.step(act)
        env.render()

        for agent in range(len(obs)):
            one_hot_action = torch.nn.functional.one_hot(act[agent], 5).squeeze().numpy()
            one_hot_agent = torch.nn.functional.one_hot(torch.tensor(agent), agent_count).numpy()
            data = {
                "obs": obs[agent].numpy(),
                "act": one_hot_action,
                "next_obs": nobs[agent],
                "rew":  rew[agent],
                "done": all(done),
                "agent": one_hot_agent,
            }
            rb.add(**data)
        
        result_map = reverse_ops(rb, vae, agent_count)
        # print(result_map)
        new_laac = torch.tensor([original_laac[result_map[agent]] for agent in range(agent_count)])
        model.laac_sample = new_laac # torch.tensor(target_laac)

        # target_laac = 1 - torch.tensor(env.agent_colors)
        target_laac = torch.tensor([original_laac[i] for i in env.shuffled_indexes])

        accuracy.append(sum([x == y for x, y in zip(new_laac, target_laac)]).item()/len(new_laac))
        # print(sum(accu), model.laac_sample, target_laac)


        # print(new_laac)

        if all(done):
            obs = [torch.from_numpy(o) for o in env.reset()]
            info["episode_reward"] = info["episode_reward"].sum()
            # random.shuffle(env.agent_colors)
            # print(env.agent_colors)
            rb.clear()
            print(info)
            print(f"Mean Episode Accuracy: {100*sum(accuracy)/len(accuracy):.2f}%")
            accuracies.append(100*sum(accuracy)/len(accuracy))

            print(f"Mean Accuracy: {sum(accuracies)/len(accuracies):.2f}%")
            accuracy = []
        else:
            obs = [torch.from_numpy(o) for o in nobs]

if __name__ == "__main__":
    main()