#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="lidar_synthesis",
    version="0.1.0",
    description="LiDAR View Synthesis for Robust Vehicle Navigation Without Expert Labels",
    author="Jonathan Schmidt",
    author_email="jonathan.schmidt@tum.de",
    url="https://github.com/jonathsch/lidar-augmentation",
    install_requires=["pytorch-lightning", "hydra-core"],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = lidar_synthesis.train:main",
        ]
    },
)
