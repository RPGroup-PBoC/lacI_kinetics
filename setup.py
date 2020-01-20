from setuptools import setup, find_packages
from os import path

__version__ = "0.0.01"

setup(
    name="lac",
    version=__version__,
    description="Various python utilities used for single-molecule analysis of LacI kinetics",
    license="BSD",
    classifiers=[
        "Development Status :: 1 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
    ],
    author="Griffin Chure",
    author_email="griffinchure@gmail.com",
)
