from setuptools import setup, find_packages

setup(
    name="pairedcl",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "torch>=1.13",
        "torchvision>=0.14",
        "gym",
    ],
)