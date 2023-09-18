# Efficient Learning of Urban Driving Policies Using Bird's-Eye-View State Representations

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

This repository is the official implementation of the [paper](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=2ttMbLQAAAAJ&citation_for_view=2ttMbLQAAAAJ:W7OEmFMy1HYC):

> **Efficient Learning of Urban Driving Policies Using Bird's-Eye-View State Representations**
>
> [Trumpp, Raphael](https://scholar.google.com/citations?user=2ttMbLQAAAAJ&hl=en), [Martin Buechner](https://rl.uni-freiburg.de/people/buechner), [Abhinav Valada](https://rl.uni-freiburg.de/people/valada), and [Marco Caccamo](https://scholar.google.com/citations?user=Jbo1MqwAAAAJ&hl=en&oi=ao).

The paper will be presented at the IEEE International Conference on Intelligent Transportation Systems 2023. If you find our work useful, please consider [citing](#reference) it.

<p align="center">
  <img src="docs/anim.gif" alt="" width="480" />
</p>

## Table of contents
- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [Reference](#reference)
- [License](#license)

## Background
Autonomous driving involves complex decision-making in highly interactive environments, requiring thoughtful negotiation with other traffic participants.
While reinforcement learning provides a way to learn such interaction behavior, efficient learning critically depends on scalable state representations.
Contrary to imitation learning methods, high-dimensional state representations still constitute a major bottleneck for deep reinforcement learning methods in autonomous driving.
In this paper, we study the challenges of constructing bird's-eye-view representations for autonomous driving and propose a recurrent learning architecture for long-horizon driving. 
Our PPO-based approach, called RecurrDriveNet, is demonstrated on a simulated autonomous driving task in CARLA, where it outperforms traditional frame-stacking methods while only requiring one million experiences for efficient training. 
RecurrDriveNet causes less than one infraction per driven kilometer by interacting safely with other road users.

### Our Contribution
Current reinforcement learning approaches for learning driving policies face the bottleneck of dimensionality. In this paper, we evaluate the efficiency of various bird's-eye-view representations used for describing the state of the driving scene. In addition to that, we propose a novel LSTM-based encoding scheme for efficiently encoding the bird's-eye-view state representation across the full trajectory of states in a reinforcement learning fashion. This alleviates the need for old-fashioned frame-stacking methods and enables further long-horizon driving research.

<p align="center">
  <img src="docs/network.png" alt="Architecture of the residual controller." width="400" />
</p>
  
### Simulator and Racetracks
This repository uses an adapted version of the  [F1TENTH gym](https://github.com/f1tenth/f1tenth_gym) as simulator.
Map data of replicated real-world racetracks are used from the [F1TENTH maps](https://github.com/f1tenth/f1tenth_racetracks) repository.

Racetracks for training and testing: (a) Nurburgring, (b) Moscow Raceway, (c) Mexico City, (d) Brands Hatch, 
(e) Sao Paulo, (f) Sepang, (g) Hockenheim, (h) Budapest, (i) Spielberg, (j) Sakhir, (k) Catalunya, and (l) Melbourne.
<p align="center">
  <img src="docs/racetracks.png" alt="Replicated real-world racetracks." width="400" />
</p>

### Results: Lap times
#### Training:
| **Track**    | **Baseline in s** | **Residual in s** | **Improvement in s** | **Improvement in %** |
|--------------|-------------------|-------------------|----------------------|----------------------|
| Nuerburgring | 60.84             | 58.07             | 2.77                 | 4.55                 |
| Moscow       | 46.75             | 43.45             | 3.30                 | 7.06                 |
| Mexico City  | 49.12             | 46.76             | 2.36                 | 4.80                 |
| Brands Hatch | 45.92             | 44.97             | 0.95                 | 2.07                 |
| Sao Paulo    | 47.92             | 44.92             | 3.00                 | 6.26                 |
| Sepang       | 66.24             | 63.18             | 3.06                 | 4.62                 |
| Hockenheim   | 49.96             | 47.35             | 2.61                 | 5.22                 |
| Budapest     | 54.33             | 51.67             | 2.66                 | 4.90                 |
| Spielberg    | 45.33             | 43.93             | 1.40                 | 3.09                 |


#### Test:
| **Track** | **Baseline in s** | **Residual in s** | **Improvement in s** | **Improvement in %** |
|-----------|-------------------|-------------------|----------------------|----------------------|
| Sakhir    | 60.34             | 57.72             | 2.62                 | 4.34                 |
| Catalunya | 56.50             | 53.54             | 1.49                 | 5.24                 |
| Melbourne | 61.03             | 59.54             | 2.96                 | 2.44                 |

#### Overall:
| **Track** | **Baseline in s** | **Residual in s** | **Improvement in s** | **Improvement in %** |
|-----------|-------------------|-------------------|----------------------|----------------------|
| Average   | 53.69             | 50.85             | 2.43                 | 4.55                 |


## Install
- We recommend to use a virtual environment for the installation:
    ```bash
    python -m venv rpl4f110_env
    source rpl4f110_env/bin/activate
    ```
- Activate the environment and install the following packages:
    ```bash
    pip install torch
    pip install gymnasium
    pip install tensorboard
    pip install hydra-core
    pip install tqdm
    pip install flatdict
    pip install torchinfo
    pip install torchrl
    pip install numba
    pip install scipy
    pip install pyglet
    pip install pillow
    pip install pyglet==1.5
    ```
- The simulator should be installed as a module:
    ```bash
    cd simulator
    pip install -e .
    ```
## Usage
### Training
After setting you desired configuration in the [config.yaml](config.yaml) file, you can start the training by running:
```bash
python main.py
```
Specific names of the experiment can be set by running:
```bash
python main.py +exp_name=your_experiment_name
```
The use of your GPU can be avoided by running:
```bash
python main.py +cuda=False
```
### Monitoring
The training results are stored in the `outputs` folder. The training progress can be monitored with tensorboard:
```bash
tensorboard --logdir outputs
```
### Others
The baseline controller can be evaluated by running:
```bash
python main.py +bench_baseline=True
```

### Docstrings
Most of the code is documented with *automatically* generated docstrings, please use them with caution.

## Reference
If you find our work useful, please consider citing our paper:

```bibtex 
@article{trumpp2023efficient,
  title={Efficient Learning of Urban Driving Policies Using Bird's-Eye-View State Representations},
  author={Trumpp, Raphael and B{\"u}chner, Martin and Valada, Abhinav and Caccamo, Marco},
  journal={arXiv preprint arXiv:2305.19904},
  year={2023}
}
```

## License
[GNU General Public License v3.0 only" (GPL-3.0)](LICENSE.txt) © [raphajaner](https://github.com/raphajaner)
