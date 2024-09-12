# Efficient Learning of Urban Driving Policies Using Bird's-Eye-View State Representations

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

This repository is the official implementation of the [paper](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=2ttMbLQAAAAJ&citation_for_view=2ttMbLQAAAAJ:W7OEmFMy1HYC):

> **Efficient Learning of Urban Driving Policies Using Bird's-Eye-View State Representations**
>
> [Trumpp, Raphael](https://scholar.google.com/citations?user=2ttMbLQAAAAJ&hl=en), [Martin Buechner](https://rl.uni-freiburg.de/people/buechner), [Abhinav Valada](https://rl.uni-freiburg.de/people/valada), and [Marco Caccamo](https://scholar.google.com/citations?user=Jbo1MqwAAAAJ&hl=en&oi=ao).

The paper will be presented at the IEEE International Conference on Intelligent Transportation Systems 2023. If you find our work useful, please consider [citing](#reference) it.

<p align="center">
  <img src="docs/teaser.png" alt="" width="400" />
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
  <img src="docs/frame_stacking.png" alt="" width="400" />
  <img width="40">
  <img src="docs/lstm.png" alt="" width="400" />
</p>
  
### Results
Based on our chosen LSTM-based encoding of bird's-eye-view representations, we achieve significantly higher average returns while reducing the number of infractions when driving compared to frame-stacking methods. This allows also robust stopping at red traffic lights.

#### Training
<p align="center">
  <img src="docs/avg_return.png" alt="Replicated real-world racetracks." width="400" />
</p>

#### Driving Behavior
<p align="center">
  <img src="docs/anim.gif" alt="" width="400" />
</p>

## Install
- We recommend to use a virtual environment for the installation:
    ```bash
    python -m venv learning2drive
    source learning2drive/bin/activate
    ```
- Activate the environment and install the following packages:
    ```bash
    pip install torch
    TBD
    ```

### Docstrings
Most of the code is documented with *automatically* generated docstrings, please use them with caution.

## Usage
TBD

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
