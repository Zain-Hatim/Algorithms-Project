from setuptools import setup, find_packages

setup(
    name="karger_stein",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "networkx",
        "numpy",
    ],
) 