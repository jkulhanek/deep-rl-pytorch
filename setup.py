from setuptools import setup

setup(
    name='deep_rl',
    version='0.1.0',
    packages=['deep_rl', 'deep_rl.a2c', 'deep_rl.common', 'deep_rl.model', 'deep_rl.deepq', 'deep_rl.a2c_unreal','deep_rl.acktr', 'deep_rl.a3c', 'deep_rl.optim'],
    author='Jonáš Kulhánek',
    author_email='jonas.kulhanek@live.com',
    license='MIT License',
    long_description=open('README.md').read(),
    install_requires = [
        'baselines @ git+https://github.com/openai/baselines.git',
        'gym'
    ]
)
