import subprocess

from setuptools import setup, find_packages
from setuptools.command.install import install

class InstallSumTreePackage(install):
    def run(self):
        install.run(self)
        subprocess.call(
            'pip install sum_tree/', shell=True
        )

setup(
    name='hanabi_agents',
    version='0.0.5',
    description='An example agent for hanabi game.',
    url='https://github.com/braintimeException/hanabi_agents',
    author='braintimeException',
    cmdclass={'install': InstallSumTreePackage},
    packages=find_packages(),
    install_requires=['optax']
)
