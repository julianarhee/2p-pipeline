#!/usr/bin/env python
from setuptools import find_packages
from setuptools import setup

VERSION = '0.1.0'

INSTALL_REQ = [
''
]

with open('README.md', 'r') as rmf:
    readme = rmf.read()

setup(
    version=VERSION,
    name='pipeline',
    author='Juliana Rhee',
    packages=find_packages(),
    author_email='rhee@g.harvard.edu',
    url="https://github.com/coxlab/2p-pipeline",
    description="Automated 2p-data processing pipeline",
    long_description=readme,
    # Installation requirements
    install_requires= INSTALL_REQ,
    data_files=[('', ['README.md'])]
)
