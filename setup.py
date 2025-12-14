from setuptools import setup, find_packages

setup(
    name='paws',
    version='1.0',
    packages=find_packages(),
    install_requires=[
    "numpy", "astropy", "pathlib", "tqdm", "scipy", "pyfstat" 
    ],
    author='Damon Cheung',
    author_email='damoncht@umich.edu',
    description='A package to manage: to create condor jobs and analysis the data for directed search of continuous gravitational wave signalsr',
    url='https://github.com/damondmc/cw_manager',  # Optional
)

