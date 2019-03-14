# Deep RL PyTorch
This repo contains implementation of popular Deep RL algorithms. Furthermore it contains unified interface for training and evaluation with unified model saving and visualization. It can be used as a good starting point when implementing new RL algorithm in PyTorch.

## Getting started
If you want to base your algorithm on this repository, start by installing it as a package
```
pip install git+https://github.com/jkulhanek/deep-rl-pytorch.git
```

If you want to run attached experiments yourself, feel free to clone this repository.
```
git clone https://github.com/jkulhanek/deep-rl-pytorch.git
```

All dependencies are prepared in a docker container. If you have nvidia-docker enabled, you can use this image. To pull and start the image just run:

```
docker run --runtime=nvidia --net=host -it kulhanek/deep-rl-pytorch:latest bash
```

From there, you can either clone your own repository containing your experiments or clone this one.

## Concepts
All algorithms are implemented as base classes. In your experiment your need to subclass from those base classes. The `deep_rl.core.AbstractTrainer` class is used for all trainers and all algorithms inherit this class. Each trainer can be wrapped in several wrappers (classes extending `deep_rl.core.AbstractWrapper`). Those wrappers are used for saving, logging, terminating the experiment and etc. All experiments should be registered using `@deep_rl.register_trainer` decorator. This decorator than wraps the trainer with default wrappers. This can be controlled by passing arguments to the decorator. All registered trainers (experiments) can be run by calling `deep_rl.make_trainer(<<name>>).run()`.

## Implemented algorithms
### A2C
A2C is a synchronous, deterministic variant of Asynchronous Advantage Actor Critic (A3C) [2] which according to OpenAI [1] gives equal performance. It is however more efficient for GPU utilization.

Start your experiment by subclassing `deep_rl.a2c.A2CTrainer`.
Several models are included in `deep_rl.a2c.model`. You may want to use at least some helper modules contained in this package when designing your own experiment.

In most of the models, initialization is done according to [3].

### Asynchronous Advantage Actor Critic (A3C) [2]
This implementation uses multiprocessing. It comes with two optimizers - RMSprop and Adam.

### Actor Critic using Kronecker-Factored Trust Region (ACKTR) [1]
This is an improvement of A2C described in [1].

## Experiments
> Comming soon

## Requirements
Those packages must be installed before using the framework for your own algorithm:
- OpenAI baselines (can be installed by running `pip install git+https://github.com/openai/baselines.git`)
- PyTorch
- Visdom (`pip install visdom`)
- Gym (`pip install gym`)
- MatPlotLib

Those packages must be installed prior running experiments:
- DeepMind Lab
- Gym[atari]

## Sources
This repository is based on work of several other authors. We would like to express our thanks.
- https://github.com/openai/baselines/tree/master/baselines
- https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/tree/master/a2c_ppo_acktr
- https://github.com/miyosuda/unreal
- https://github.com/openai/gym

## References
[1] Wu, Y., Mansimov, E., Grosse, R.B., Liao, S. and Ba, J., 2017. Scalable trust-region method for deep reinforcement learning using kronecker-factored approximation. In Advances in neural information processing systems (pp. 5279-5288).

[2] Mnih, V., Badia, A.P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., Silver, D. and Kavukcuoglu, K., 2016, June. Asynchronous methods for deep reinforcement learning. In International conference on machine learning (pp. 1928-1937).

[3] Saxe, A.M., McClelland, J.L. and Ganguli, S., 2013. Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. arXiv preprint arXiv:1312.6120.

