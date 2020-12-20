from setuptools import setup, find_packages
from deep_rl import __version__

setup(
    name='deep_rl',
    version=__version__,
    packages=find_packages(include=('deep_rl', 'deep_rl.*')),
    author='Jonáš Kulhánek',
    author_email='jonas.kulhanek@live.com',
    license='MIT License',
    long_description=open('README.md').read(),
    install_requires=[x.rstrip('\n') for x in open('requirements.txt')])
