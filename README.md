# Robust Deep Reinforcement Learning through Adversarial Loss

This repository is the official implementation of [Robust Deep Reinforcement Learning through Adversarial Loss](https://arxiv.org/abs/2030.12345). (link is a placeholder)
Our code builds on top of various existing implementations, most notably:

A3C implementation and overall flow based on https://github.com/dgriff777/rl_a3c_pytorch.

DQN implementation based on https://github.com/higgsfield/RL-Adventure

Adversarial attack implementations based on https://github.com/advboxes/AdvBox/blob/master/advbox.md.



## Requirements
To run our code you need to have Python 3 (>=3.7) and pip installed on your systems. Additionally we require PyTorch>=1.4, which should be installed using instructions from https://pytorch.org/get-started/locally/.

To install requirements:

```setup
pip install -r requirements.txt
```

## Pre-trained Models

You can download our trained models for DQN here:(tbd), and for A3C here:(tbd). We suggest unpacking these to 'DQN/trained_models/' and 'A3C/trained_models/' respectively.

## Training

To train a standard DQN model on Pong like the one used in our paper, run this command:

```train DQN
cd DQN
python main.py 
```
To speed up training by using gpu x (in a system with one gpu x=0) add the following argument --gpu-id x.
To train in another game, like RoadRunner use --env RoadRunnerNoFrameskip-v4. Other games used in the paper are FreewayNoFrameskip-v4 and BankHeistNoFrameskip-v4. 

```train A3C
cd A3C
python main.py 
```
Additionally you can use --gpu-ids argument to train with one or more gpus, for example use GPUs 0 and 1 with '--gpu-ids 0 1'. Note the default value of workers used for A3C is 16, and you might want to change it to the amount of cpu cores in system for max efficiency with the argument --workers 4 for example. This may effect results however.

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
Additionally you can use --gpu-id x argument to use a GPU to speed up evaluation. Note that pgd takes much longer to run than other evaluation metrics, so you can try replacing it with much faster evaluation against FGSM attacks by switching the command to --fgsm.

Results will be saved in numpy arrays, and the result_viewer.ipynb provide a convenient way to view them.


## Results

TODO: fill this

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

> 📋Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

> 📋Pick a licence and describe how to contribute to your code repository. 
