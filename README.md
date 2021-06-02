# Selective Parameter Sharing
Official implementation of the ICML 2021 paper [Scaling Multi-Agent Reinforcement Learning with Selective Parameter Sharing](https://arxiv.org/abs/2102.07475)

## Requirements

To run this repository you need:
i) To install the code's requirements. A virtual environment is recommended. Install with 
```setup
pip install -r requirements.txt
```
ii) Install at least one of the supported environments:
- Multi-Robot Warehouse found [here](https://github.com/semitable/robotic-warehouse/tree/color) (SePS requires the `color` branch of RWARE which supports multiple agent types) 
- Blind-Particle Spread from our fork of Petting-Zoo [here](https://github.com/semitable/PettingZoo)
- Level-Based Foraging found [here](https://github.com/semitable/robotic-warehouse) (Important! LBF must be modified to make agent levels persist across environment resets)
- Starcraft Multi-Agent Challenge [here](https://github.com/oxwhirl/smac) (Careful! Requires installation of SC2 which is a large package. For our implementation you _must_ remove that SMACCompatible wrapper from the configuration, by removing the respective line in `ac.py:config`)

## Running

You can then simply run ac.py using:
```python
    python ac.py with env_name='robotic_warehouse:rware-2color-tiny-4ag-v1' time_limit=500
```
The above command will load and run the tiny variation of the RWARE task with four agents. Other exciting environments shown in the paper include 'rware-2color-tiny4-ag-hard-v1', 
'rware-2color-small-8ag-v1' or 'rware-2color-medium-16ag-v1' (replace in "robotic_warehouse:XXXX" above). Episode time limit should be kept at 500 as above.

The blind particle spread (BPS) environment can be run using:
```python
    python ac.py with env_name='pettingzoo:pz-mpe-large-spread-v1' time_limit=50
```
In the paper, we have used an episode time limit of 50. `v1`-`v6` contain different scenarios (number of agents/landmarks etc) as described in the paper.

Similarly, LBF and SMAC can be run as long as they are registered in gym.

## A note on the hyperparameters

This repository is an implementation of SePS and not necessarily the identical code that generated the experiments (i.e. small bug-fixes, features, or improved hyperparameters _will_ be contained in this repository). Any major bug fixes (if necessary) will be documented below. 

Right now, the proposed hyperparameters can be found in the config function of `ops_utils.py` (for the SePS procedure) and for `ac.py` (A2C hyperparameters).
For SePS (`ops_utils.py`):
- `pretraining_steps`: always returns stabler results when increased. Usually works with as low as 2,000, but should be increased to 10,000 if time/memory allows. Simpler environments (like BPS) can handle much smaller values.
- `clusters`: the number of clusters (K in the paper). Please see respective section in the paper. When # of clusters is unknown Davies–Bouldin index is recommended (scikit-learn function [here](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.davies_bouldin_score.html) works well). DB-Index will be used automatically if `clusters=None`.
- `z_features`: Typical values of ~5-10 work well with the tested number of types. If this value is set to 2 then a visualisation will be saved at "clusters.png"
- `kl_weight`: Values near zero work well since it tends to overwhelm the reconstruction loss. Try setting to zero when debugging.
- `reconstruct`: Could be changed from `["next_obs", "rew"]` to only `["next_obs"]` or `["rew"]` if it known that the environment only changes the observation or reward function respectively (and not both).
- `delay/delay_training/pretraining_times`: can be used in situation when differences between types are shown later in the training.

In `ac.py`:
- "algorithm_mode": Can be set to "ops", "iac", "snac-a", "snac-b". These values correspond to "SePS", "NoPS", "FuPS", "FuPS+id" respectively.

## model.py

`model.py` contains a fast implementation of selective parameter sharing for multi-agent RL. It can be used independently from the SePS codebase.  
`MultiAgentFCNetwork` defines a torch network with K independent networks. By passing a list (size>K) of inputs and a `laac_index` that maps the index of the input to the index of the independent network, it will quickly resolve them and return the desired returns. Example: if a K of 2 is selected and you have four agents, you can pass a laac_index of `[0, 0, 1, 1]` to share parameters between the first two agents and the last two agents.

# Cite:

```
@inproceedings{christianos2021scaling,
   title={Scaling Multi-Agent Reinforcement Learning with Selective Parameter Sharing},
   author={Filippos Christianos and Georgios Papoudakis and Arrasy Rahman and Stefano V. Albrecht},
   booktitle={International Conference on Machine Learning (ICML)},
   year={2021}
}
```
