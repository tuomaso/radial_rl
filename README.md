# Robust Deep Reinforcement Learning through Adversarial Loss

This repository is the official implementation of [Robust Deep Reinforcement Learning through Adversarial Loss](https://arxiv.org/abs/2030.12345). (link is a placeholder)

# Overview

**RADIAL**(**R**obust **AD**versar**IA**l **L**oss) - RL, is a framework for training more robust deep RL agents. It leverages algorithms for calculating certified output bounds such as Interval Bound Probagation to minimize an upper bound of the original algorithms loss function under worst possible (bounded) adversarial perturbation. This framework significantly increases neural net robustness against PGD-attacks. 
In addition we propose *Greedy Worst-Case Reward (GWC)*, an efficient method for estimating agents performance under worst possible sequence of adversarial attacks.


Our code builds on top of various existing implementations, most notably:

* A3C implementation and overall flow based on https://github.com/dgriff777/rl_a3c_pytorch.
* DQN implementation based on https://github.com/higgsfield/RL-Adventure
* Adversarial attack implementations based on https://github.com/advboxes/AdvBox/blob/master/advbox.md.



## Requirements
To run our code you need to have Python 3 (>=3.7) and pip installed on your systems. Additionally we require PyTorch>=1.4, which should be installed using instructions from https://pytorch.org/get-started/locally/.

To install requirements:

```setup
pip install -r requirements.txt
```

## Pre-trained Models

You can download our trained models for DQN here:(tbd), and for A3C here:(tbd). We suggest unpacking these to `radial_rl/DQN/trained_models/` and `radial_rl/A3C/trained_models/` respectively.

## Training

To train a standard DQN model on Pong like the one used in our paper, run this command:

```train DQN
cd DQN
python main.py 
```
To speed up training by using gpu x (in a system with one gpu x=0) add the following argument `--gpu-id x`.
To train in another game, like RoadRunner use `--env RoadRunnerNoFrameskip-v4`. Other games used in the paper are `FreewayNoFrameskip-v4 `and `BankHeistNoFrameskip-v4`. 

```train A3C
cd A3C
python main.py 
```
Additionally you can use --gpu-ids argument to train with one or more gpus, for example use GPUs 0 and 1 with `--gpu-ids 0 1`. Note the default value of workers used for A3C is 16, and you might want to change it to the amount of cpu cores in system for max efficiency with the argument `--workers 4` for example. This may effect results however.

The models will be saved in args.save_model_dir, with a name of their environment and time and date training started. Each run produces two models but we used the \_last.pt for all experiments, while \_best.pt is mostly useful as intermediate checkpoint if training is disrupted. 


## Robust training

To train a robust DQN model on RoadRunner like the one used in our paper, using our pre-trained RoadRunner model, use the following:

```Radial DQN
cd DQN
python main.py --env RoadRunnerNoFrameskip-v4 --robust --load-path "trained_models/RoadRunnerNoFrameskip-v4_trained.pt" --total-frames 4500000 --exp-epsilon-decay 1 --replay-initial 256
```


```Radial A3C
cd A3C
python main.py --env RoadRunnerNoFrameskip-v4 --robust --load-path "trained_models/RoadRunnerNoFrameskip-v4_trained.pt" --total-frames 10000000
```



## Evaluation

To evaluate our robustly trained BankHeist model using the metrics described in the paper, use the following command in the DQN or A3C directory:

```
python evaluate.py --env BankHeistNoFrameskip-v4 --load-path "trained_models/BankHeistNoFrameskip-v4_robust.pt" --pgd --gwc --nominal 
```
Additionally you can use `--gpu-id x` argument to use a GPU to speed up evaluation. Note that pgd takes much longer to run than other evaluation metrics, so you can try replacing it with much faster evaluation against FGSM attacks by switching the command to `--fgsm`.

Results will be saved in numpy arrays, and the result_viewer.ipynb provide a convenient way to view them.


## Results

### Robustness on Atari games
| Game         | Model\Metric |  PGD attack  |             |  GWC reward  |
|--------------|--------------|:------------:|:-----------:|:------------:|
|              |    epsilon   |     1/255    |    3/255    |     1/255    |
|     Pong     | RS-DQN       |     18.13    |      -      |       -      |
|              | SA-DQN       |   20.1+-0.0  |      -      |       -      |
|              | RADIAL-DQN   |  20.8+-0.09  |  20.8+-0.09 |  -1.85+-4.62 |
|              | RADIAL-A3C   |   20.0+-0.0  |  20.0+-0.0  |   20.0+-0.0  |
|    Freeway   | RS-DQN       |     32.53    |      -      |       -      |
|              | SA-DQN       |  30.36+-0.7  |      -      |       -      |
|              | RADIAL-DQN   |  21.95+-0.40 | 21.55+-0.26 |  21.7+-0.39  |
|   BankHeist  | RS-DQN       |    190.67    |      -      |       -      |
|              | SA-DQN       |  1043.6+-9.5 |      -      |       -      |
|              | RADIAL-DQN   | 1038.0+-23.0 | 833.5+-45.2 | 1048.0+-32.3 |
|              | RADIAL-A3C   |  848.0+-3.8  |  827.0+-6.0 |  832.5+-4.1  |
|  RoadRunner  | RS-DQN       |    5753.33   |      -      |       -      |
|              | SA-DQN       |  15280+-828  |      -      |       -      |
|              | RADIAL-DQN   |  43920+-1238 |  12480+-901 |  33745+-2389 |
|              | RADIAL-A3C   |  30435+-1504 | 30620+-1141 |  29595+-1428 |

### Training commands for models above
For DQN models make sure you are in the `radial_rl/DQN` directory before issuing commands, and in the `radial_rl/A3C` directory for A3C models. And have downloaded the pretrained models to specified directories.

| Game       | Model      |                                                                                          Command                                                                                          |
|------------|------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|    Pong    | RADIAL-DQN |               python main.py --robust --load-path "trained_models/PongNoFrameskip-v4_trained.pt" --total-frames 4500000 --exp-epsilon-decay 1 --replay-initial 256 --amsgrad              |
|            | RADIAL-A3C |                                         python main.py --robust --load-path "trained_models/PongNoFrameskip-v4_trained.pt" --total-frames 10000000                                        |
|   Freeway  | RADIAL-DQN |    python main.py --env FreewayNoFrameskip-v4 --robust --load-path "trained_models/FreewayNoFrameskip-v4_trained.pt" --total-frames 4500000 --exp-epsilon-decay 1 --replay-initial 256    |
|  BankHeist | RADIAL-DQN |  python main.py --env BankHeistNoFrameskip-v4 --robust --load-path "trained_models/BankHeistNoFrameskip-v4_trained.pt" --total-frames 4500000 --exp-epsilon-decay 1 --replay-initial 256  |
|            | RADIAL-A3C |                       python main.py --env BankHeistNoFrameskip-v4 --robust --load-path "trained_models/BankHeistNoFrameskip-v4_trained.pt" --total-frames 10000000                       |
| RoadRunner | RADIAL-DQN | python main.py --env RoadRunnerNoFrameskip-v4 --robust --load-path "trained_models/RoadRunnerNoFrameskip-v4_trained.pt" --total-frames 4500000 --exp-epsilon-decay 1 --replay-initial 256 |
|            | RADIAL-A3C |                      python main.py --env RoadRunnerNoFrameskip-v4 --robust --load-path "trained_models/RoadRunnerNoFrameskip-v4_trained.pt" --total-frames 10000000

## Contributing
This code version is for NeurIPS 2020 review only, do not distribute.

## Common issues

On some machines you might get the following error ImportError: libSM.so.6: cannot open shared object file: No such file or directory,
which can be fixed by running the following line: 
```
sudo apt-get install libsm6 libxrender1 libfontconfig1
```
