#!/usr/bin/env python
# -*- coding: utf-8 -*-

import setuptools


with open('README.md', 'r', encoding='utf-8') as f:
    README = f.read()

setuptools.setup(
    name='PyTorch-DRL',
    version='0.0.1',
    description='A simple PyTorch wrapper making deep reinforcement learning much easier',
    author='Yi Zhang',
    author_email='yizhang.dev@gmail.com',
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://github.com/imyizhang/PyTorch-DRL',
    download_url='https://github.com/imyizhang/PyTorch-DRL',
    packages=setuptools.find_packages(),
    keywords=[
        'pytorch', 'reinforcement-learning', 'drl', 'pytorch-drl',
    ],
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
    ],
    license='MIT',
    python_requires='>=3.8',
    install_requires=[
        'torch>=1.8.1'
    ],
)
