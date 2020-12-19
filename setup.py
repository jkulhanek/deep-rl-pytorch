from setuptools import setup
from deep_rl import VERSION

setup(
    name='deep_rl',
    version=VERSION,
    packages=['deep_rl', 'deep_rl.actor_critic', 'deep_rl.actor_critic.unreal', 'deep_rl.common', 'deep_rl.model', 'deep_rl.deepq', 'deep_rl.optim'],
    author='Jonáš Kulhánek',
    author_email='jonas.kulhanek@live.com',
    license='MIT License',
    long_description=open('README.md').read(),
    install_requires=[
        #'baselines @ git+https://github.com/openai/baselines.git',
        'torch',
        'gym'
    ]
)
