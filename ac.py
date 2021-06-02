import glob
import logging
import os
import shutil
import time
from collections import deque
from functools import partial
from os import path
from pathlib import Path
from collections import defaultdict

import pickle
from cpprb import ReplayBuffer, create_before_add_func, create_env_dict
import gym
import numpy as np
import torch
from sacred import Experiment
from sacred.observers import (
    FileStorageObserver,
    MongoObserver,  # noqa
    QueuedMongoObserver,
    QueueObserver,
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnvWrapper
from torch.utils.tensorboard import SummaryWriter

from model import Policy
from ops_utils import compute_clusters, ops_ingredient
from wrappers import *

ex = Experiment(ingredients=[ops_ingredient])
# ex.captured_out_filter = apply_backspaces_and_linefeeds
ex.captured_out_filter = lambda captured_output: "Output capturing turned off."
ex.observers.append(FileStorageObserver('output/'))

logging.basicConfig(
    level=logging.INFO,
    format="(%(process)d) [%(levelname).1s] - (%(asctime)s) >> %(message)s",
    datefmt="%m/%d %H:%M:%S",
)


@ex.config
def config(ops):
    name = "SePS release"
    version = 0

    env_name = None
    time_limit = None
    env_args = {}

    wrappers = (
        RecordEpisodeStatistics,
        SquashDones,
        SMACCompatible,
    )
    dummy_vecenv = True

    # everything below is update steps (not env steps!)
    total_steps = int(10e6)
    log_interval = int(2e3)
    save_interval = int(1e6)
    eval_interval = int(1e4)

    architecture = {
        "actor": [64, 64],
        "critic": [64, 64],
    }

    lr = 3e-4
    optim_eps = 0.00001

    parallel_envs = 8
    n_steps = 5
    gamma = 0.99
    entropy_coef = 0.01
    value_loss_coef = 0.5
    use_proper_termination = True
    central_v = False

    # 
    algorithm_mode = "ops" # "ops", "iac", "snac", or "snac-a"

    #
    device = "cpu"


class Torcherize(VecEnvWrapper):
    @ex.capture
    def __init__(self, venv, algorithm_mode):
        super().__init__(venv)
        self.observe_agent_id = algorithm_mode == "snac-a"
        if self.observe_agent_id:
            agent_count = len(self.observation_space)
            self.observation_space = gym.spaces.Tuple(tuple([gym.spaces.Box(low=-np.inf, high=np.inf, shape=((x.shape[0] + agent_count),), dtype=x.dtype) for x in self.observation_space]))

    @ex.capture
    def reset(self, device, parallel_envs):
        obs = self.venv.reset()
        obs = [torch.from_numpy(o).to(device) for o in obs]
        if self.observe_agent_id:
            ids = torch.eye(len(obs)).repeat_interleave(parallel_envs, 0).view(len(obs), -1, len(obs))
            obs = [torch.cat((ids[i], obs[i]), dim=1) for i in range(len(obs))]
        return obs

    def step_async(self, actions):
        actions = [a.squeeze().cpu().numpy() for a in actions]
        actions = list(zip(*actions))
        return self.venv.step_async(actions)

    @ex.capture
    def step_wait(self, device, parallel_envs):
        obs, rew, done, info = self.venv.step_wait()
        obs = [torch.from_numpy(o).float().to(device) for o in obs]
        if self.observe_agent_id:
            ids = torch.eye(len(obs)).repeat_interleave(parallel_envs, 0).view(len(obs), -1, len(obs))
            obs = [torch.cat((ids[i], obs[i]), dim=1) for i in range(len(obs))]

        return (
            obs,
            torch.from_numpy(rew).float().to(device),
            torch.from_numpy(done).float().to(device),
            info,
        )


class SMACWrapper(VecEnvWrapper):
    def _make_action_mask(self, n_agents):
        action_mask = self.venv.env_method("get_avail_actions")
        action_mask = [
            torch.tensor([avail[i] for avail in action_mask]) for i in range(n_agents)
        ]
        return action_mask

    def _make_state(self, n_agents):
        state = self.venv.env_method("get_state")
        state = torch.from_numpy(np.stack(state))
        return n_agents * [state]

    def reset(self):
        obs = self.venv.reset()
        state = self._make_state(len(obs))
        action_mask = self._make_action_mask(len(obs))
        return obs, state, action_mask

    def step_wait(self):
        obs, rew, done, info = self.venv.step_wait()
        state = self._make_state(len(obs))
        action_mask = self._make_action_mask(len(obs))

        return (
            (obs, state, action_mask),
            rew,
            done,
            info,
        )


@ex.capture
def _compute_returns(storage, next_value, gamma):
    returns = [next_value]
    for rew, done in zip(reversed(storage["rewards"]), reversed(storage["done"])):
        ret = returns[0] * gamma + rew * (1 - done.unsqueeze(1))
        returns.insert(0, ret)

    return returns


@ex.capture
def _make_envs(env_name, env_args, parallel_envs, dummy_vecenv, wrappers, time_limit, seed):
    def _env_thunk(seed):
        # print(env_args)
        env = gym.make(env_name, **env_args)
        if time_limit:
            env = TimeLimit(env, time_limit)
        for wrapper in wrappers:
            env = wrapper(env)
        env.seed(seed)
        return env

    env_thunks = [partial(_env_thunk, seed + i) for i in range(parallel_envs)]
    if dummy_vecenv:
        envs = DummyVecEnv(env_thunks)
        envs.buf_rews = np.zeros(
            (parallel_envs, len(envs.observation_space)), dtype=np.float32
        )
    else:
        envs = SubprocVecEnv(env_thunks, start_method="fork")
    envs = Torcherize(envs)
    envs = SMACWrapper(envs)
    return envs


def _squash_info(info):
    info = [i for i in info if i]
    new_info = {}
    keys = set([k for i in info for k in i.keys()])
    keys.discard("TimeLimit.truncated")
    for key in keys:
        mean = np.mean([np.array(d[key]).sum() for d in info if key in d])
        new_info[key] = mean
    return new_info


@ex.capture
def _log_progress(
    infos,
    prev_time,
    step,
    parallel_envs,
    n_steps,
    total_steps,
    log_interval,
    _log,
    _run,
):

    elapsed = time.time() - prev_time
    ups = log_interval / elapsed
    fps = ups * parallel_envs * n_steps
    mean_reward = sum(sum([ep["episode_reward"] for ep in infos]) / len(infos))
    battles_won = 100 * sum([ep.get("battle_won", 0) for ep in infos]) / len(infos)

    _log.info(f"Updates {step}, Environment timesteps {parallel_envs* n_steps * step}")
    _log.info(
        f"UPS: {ups:.1f}, FPS: {fps:.1f}, ({100*step/total_steps:.2f}% completed)"
    )

    _log.info(f"Last {len(infos)} episodes with mean reward: {mean_reward:.3f}")
    _log.info(f"Battles won: {battles_won:.1f}%")
    _log.info("-------------------------------------------")

    squashed_info = _squash_info(infos)
    for k, v in squashed_info.items():
        _run.log_scalar(k, v, step)


@ex.capture
def _compute_loss(model, storage, value_loss_coef, entropy_coef, central_v):
    with torch.no_grad():
        next_value = model.get_value(storage["state" if central_v else "obs"][-1])
    returns = _compute_returns(storage, next_value)

    input_obs = zip(*storage["obs"])
    input_obs = [torch.stack(o)[:-1] for o in input_obs]

    if central_v:
        input_state = zip(*storage["state"])
        input_state = [torch.stack(s)[:-1] for s in input_state]
    else:
        input_state = None

    input_action_mask = zip(*storage["action_mask"])
    input_action_mask = [torch.stack(a)[:-1] for a in input_action_mask]

    input_actions = zip(*storage["actions"])
    input_actions = [torch.stack(a) for a in input_actions]

    values, action_log_probs, entropy = model.evaluate_actions(
        input_obs, input_actions, input_action_mask, input_state,
    )

    returns = torch.stack(returns)[:-1]
    advantage = returns - values

    actor_loss = (
        -(action_log_probs * advantage.detach()).sum(dim=2).mean()
        - entropy_coef * entropy
    )
    value_loss = (returns - values).pow(2).sum(dim=2).mean()

    loss = actor_loss + value_loss_coef * value_loss
    return loss

@ex.automain
def main(
    _run,
    seed,
    total_steps,
    log_interval,
    save_interval,
    eval_interval,
    architecture,
    lr,
    optim_eps,
    parallel_envs,
    n_steps,
    use_proper_termination,
    central_v,

    ops,
    algorithm_mode,
    env_name,

    device,
    _log,
):
    torch.set_num_threads(1)

    envs = _make_envs()

    agent_count = len(envs.observation_space)
    obs_size = envs.observation_space[0].shape
    act_size = envs.action_space[0].n

    env_dict = {
        "obs": {"shape": obs_size, "dtype": np.float32},
        "rew": {"shape": 1, "dtype": np.float32},
        "next_obs": {"shape": obs_size, "dtype": np.float32},
        "done": {"shape": 1, "dtype": np.float32},
        "act": {"shape": act_size, "dtype": np.float32},
        "agent": {"shape": agent_count, "dtype": np.float32},
    }
    rb = ReplayBuffer(int(agent_count * ops['pretraining_steps'] * parallel_envs * n_steps), env_dict)

    # before_add = create_before_add_func(env)

    state_size = envs.get_attr("state_size")[0] if central_v else None
    
    if algorithm_mode.startswith("snac"):
        model_count = 1
    elif algorithm_mode == "iac":
        model_count = len(envs.observation_space)
    elif algorithm_mode == "ops":
        if ops["clusters"]:
            model_count = ops["clusters"]
        else:
            model_count = min(10, len(envs.observation_space))

    # make actor-critic model
    model = Policy(envs.observation_space, envs.action_space, architecture, model_count, state_size)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr, eps=optim_eps)

    # creates and initialises storage
    obs, state, action_mask = envs.reset()

    storage = defaultdict(lambda: deque(maxlen=n_steps))
    storage["obs"] = deque(maxlen=n_steps + 1)
    storage["done"] = deque(maxlen=n_steps + 1)
    storage["obs"].append(obs)
    storage["done"].append(torch.zeros(parallel_envs))
    storage["info"] = deque(maxlen=10)

    # for smac:
    storage["state"] = deque(maxlen=n_steps + 1)
    storage["action_mask"] = deque(maxlen=n_steps + 1)
    if central_v:
        storage["state"].append(state)
    storage["action_mask"].append(action_mask)
    # ---------

    model.sample_laac(parallel_envs)
    if algorithm_mode == "iac":
        model.laac_sample = torch.arange(len(envs.observation_space)).repeat(parallel_envs, 1)
        # print(model.laac_sample)
    if algorithm_mode == "ops":
        model.laac_sample = torch.zeros(parallel_envs, agent_count).long()
        # print(model.laac_sample)

    start_time = time.time()
    for step in range(total_steps):
        
        if algorithm_mode == "ops" and step in [ops["delay"] + ops["pretraining_steps"]*(i+1) for i in range(ops["pretraining_times"])]:
            print(f"Pretraining at step: {step}")
            cluster_idx = compute_clusters(rb.get_all_transitions(), agent_count)
            model.laac_sample = cluster_idx.repeat(parallel_envs, 1)
            pickle.dump(rb.get_all_transitions(), open(f"{env_name}.p", "wb"))
            _log.info(model.laac_sample)


        if step % log_interval == 0 and len(storage["info"]):
            _log_progress(storage["info"], start_time, step)
            start_time = time.time()
            storage["info"].clear()

        for n_step in range(n_steps):
            with torch.no_grad():
                actions = model.act(storage["obs"][-1], storage["action_mask"][-1])
            (obs, state, action_mask), reward, done, info = envs.step(actions)

            if use_proper_termination:
                bad_done = torch.FloatTensor(
                    [1.0 if i.get("TimeLimit.truncated", False) else 0.0 for i in info]
                ).to(device)
                done = done - bad_done

            storage["obs"].append(obs)
            storage["actions"].append(actions)
            storage["rewards"].append(reward)
            storage["done"].append(done)
            storage["info"].extend([i for i in info if "episode_reward" in i])
            storage["laac_rewards"] += reward

            if algorithm_mode == "ops" and step < ops["delay"] + ops["pretraining_times"] * ops["pretraining_steps"]:
                for agent in range(len(obs)):

                    one_hot_action = torch.nn.functional.one_hot(actions[agent], act_size).squeeze().numpy()
                    one_hot_agent = torch.nn.functional.one_hot(torch.tensor(agent), agent_count).repeat(parallel_envs, 1).numpy()

                    if bad_done[0]:
                        nobs = info[0]["terminal_observation"]
                        nobs = [torch.tensor(o).unsqueeze(0) for o in nobs]
                    else:
                        nobs = obs
                        
                    data = {
                        "obs": storage["obs"][-2][agent].numpy(),
                        "act": one_hot_action,
                        "next_obs": nobs[agent].numpy(),
                        "rew":  reward[:, agent].unsqueeze(-1).numpy(),
                        "done": done[:].unsqueeze(-1).numpy(),
                        # "policy": np.array([model.laac_sample[0, agent].float().item()]),
                        "agent": one_hot_agent,
                        # "timestep": step,
                        # "nstep": n_step,
                    }
                    rb.add(**data)

            # for smac:
            if central_v:
                storage["state"].append(state)

            storage["action_mask"].append(action_mask)
            # ---------

        if algorithm_mode == "ops" and step < ops["pretraining_steps"] and ops["delay_training"]:
            continue

        loss = _compute_loss(model, storage)

        # if laac_mode=="laac" and step and step % laac_timestep == 0:
        #     loss += _compute_laac_loss(model, storage)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

    # df["agent"] = df["agent"].astype(int)
    # df["timestep"] = df["timestep"].astype(int)
    # df = df.set_index(["timestep", "agent"])

    envs.close()