# deepdrivewe-academy
Academy implementation of DeepDriveWE.

Implementation of [DeepDriveWE](https://pubs.acs.org/doi/full/10.1021/acs.jctc.4c01136) using [Academy](https://docs.academy-agents.org/stable/).

## Installation

To install the package, run the following command:
```bash
git clone git@github.com:braceal/deepdrivewe-academy.git
cd deepdrivewe-academy
pip install -e .
```

Full installation including dependencies:
```bash
git clone git@github.com:braceal/deepdrivewe-academy.git
cd deepdrivewe-academy
conda create -n deepdrivewe python=3.10 -y
conda install omnia::ambertools -y
conda install conda-forge::openmm==7.7 -y
pip install -e .
```

To use deep learning models, install the correct version of [PyTorch](https://pytorch.org/get-started/locally/)
for your system and drivers. To use `mdlearn`, you may need an earlier version of PyTorch:
```bash
pip install torch==1.12
```

## Contributing

For development, it is recommended to use a virtual environment. The following
commands will create a virtual environment, install the package in editable
mode, and install the pre-commit hooks.
```bash
python -m venv venv
source venv/bin/activate
pip install -U pip setuptools wheel
pip install -e '.[dev,docs]'
pre-commit install
```
To test the code, run the following command:
```bash
pre-commit run --all-files
tox -e py310
```
