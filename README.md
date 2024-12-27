# Interpreting Emergent Planning in Model-Free Reinforcement Learning

This is the official repo for the paper [*Interpreting Emergent Planning in Model-Free Reinforcement Learning*](https://openreview.net/forum?id=DzGe40glxs&referrer=%5BAuthor%20Console%5D). A blog post summarising this paper can be found [here](https://tuphs28.github.io/). This repo builds on the [*Thinker*](https://github.com/stephen-chung-mh/thinker) repo associated with the paper [*Thinker: Learning to Plan and Act*](https://arxiv.org/abs/2307.14993).

## Table of Contents
- [Installation](#installation)

##  Installation
1. Update essential packages and install Cython:

```bash
sudo apt-get update
sudo apt-get install zip python-opencv build-essential -y
pip install Cython
```

2. Compile and install Sokoban:
```bash
cd sokoban
pip install -e .
```

3. Compile and install Thinker:
```bash
cd thinker
pip install -e .
```

4. Install remaining packages used in probing experiments
```bash
cd experiments
pip install -r requirements.txt
```
## Sokoban Experiments

All code to reproduce the interpretability results on Sokoban can be found in the `experiments/sokoban_experiments` directory.

### Training Linear Probes

Before training a probe, we must generate a probing training and a test dataset. This can be done using the following commands
```bash
python3 create_probe_dataset.py --model_name "250m" --num_episodes 3000 --name "train"
python3 create_probe_dataset.py --model_name "250m" --num_episodes 1000 --name "test" --env_name "valid-"
```
