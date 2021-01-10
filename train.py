#!/usr/bin/env python
import argparse
from deep_rl import make_trainer, configure
from deep_rl.common.metrics import MetricHandlerBase
import torch.multiprocessing as mp


class WandbMetricHandler(MetricHandlerBase):
    def __init__(self, *args, **kwargs):
        self.run = None
        super(WandbMetricHandler, self).__init__("wandb", *args, **kwargs)

    def collect(self, collection, time, mode='train'):
        if self.run is None:
            import wandb
            self.run = wandb.init(project='deep-rl-pytorch')
        self.run.log(dict(collection), step=time)
        pass


if __name__ == '__main__':
    configure(
        logging=dict(
            handlers=[
                "file",
                "matplotlib",
                WandbMetricHandler
            ]
        ),
    )

    # Set mp method to spawn
    # Fork does not play well with pytorch
    mp.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str, help='Experiment name')
    args = parser.parse_args()
    name = args.name

    package_name = 'experiments.%s' % name.replace('-', '_')
    package = __import__(package_name, fromlist=[''])
    default_args = package.default_args

    trainer = make_trainer(name, **default_args())
    trainer.run()
