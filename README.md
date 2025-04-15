A repo for my reinforcement learning projects.

# How to install Atari environments

First we need to install `gymnasium` and `ale-py`. I use [mamba](https://github.com/mamba-org/mamba) to manage the virtual environment. For conda users, you can use the conda command to install the packages.

```bash
mamba install conda-forge::gymnasium
mamba install -c conda-forge gymnasium-atari
mamba install conda-forge::ale-py
```

Then we I run the following code,

```python
import gymnasium as gym
import ale_py

env = gym.make("ALE/Pong-v5")
```

I got the following error:

```
FileNotFoundError: [Errno 2] No such file or directory: '/home/liyuan/miniforge3/envs/transformer2/lib/python3.12/site-packages/ale_py/roms/pong.bin'
```

To solve the issue, we need to download the rom files with [AutoRom](https://pypi.org/project/AutoROM/)

Install AutoRom
```bash
pip install autorom
```

Then download the rom files to the target directory

```bash
AutoROM --install-dir /home/liyuan/miniforge3/envs/transformer2/lib/python3.12/site-packages/ale_py/roms
```
