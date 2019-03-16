import argparse
from deep_rl import make_trainer, configure
import torch.multiprocessing as mp

if __name__ == '__main__':
    configure(
        visdom=dict(
            server='http://localhost',
            port=8097
        )
    )

    # Set mp method to spawn
    # Fork does not play well with pytorch
    mp.set_start_method('spawn')


    parser = argparse.ArgumentParser()
    parser.add_argument('name', type = str, help = 'Experiment name')
    args = parser.parse_args()
    name = args.name


    package_name = 'experiments.%s' % name.replace('-', '_')
    package = __import__(package_name, fromlist=[''])
    default_args = package.default_args
        
    trainer = make_trainer(name, **default_args())
    trainer.run()