# Installation


## Prerequisites

### Python 3.7+ environment

The use of a virtual environment is recommended, and you will need to ensure that the environment use a Python version
greater than 3.7.
This can be achieved for instance either by using [conda](https://docs.conda.io/en/latest/) or by using [pyenv](https://github.com/pyenv/pyenv) (or [pyenv-win](https://github.com/pyenv-win/pyenv-win) on windows)
and [venv](https://docs.python.org/fr/3/library/venv.html) module.

The following examples show how to create a virtual environment with Python version 3.8.13 with the mentioned methods.

#### With conda (all platforms)

```shell
conda create -n do-env python=3.8.13
conda activate do-env
```

#### With pyenv + venv (Linux/MacOS)

```shell
pyenv install 3.8.13
pyenv shell 3.8.13
python -m venv do-venv
source do-venv/bin/activate
```

#### With pyenv-win + venv (Windows)

```shell
pyenv install 3.8.13
pyenv shell 3.8.13
python -m venv do-venv
do-venv\Scripts\activate
```

### Tensorflow 2.6+

A version of tensorflow greater than 2.6 is required. If no package tensorflow is detected when installing decomon, a package tensorflow will be installed.
Howver it is better to do it beforehand to ensure having a version that is adapted to your GPU/CPU. See [tensorflow documentation](https://www.tensorflow.org/install/pip)
to see how to install it properly.


## Pip install decomon library

Install decomon from pip:

```shell
pip install decomon
```

**NB**: If you want to install the latest version available on github repository, you can do

```shell
pip install git+https://github.com/airbus/decomon@main#egg=decomon
```

## Issues

If you have any issue when installing, you may need to update pip and setuptools:

```shell
pip install --upgrade pip setuptools
```

If still not working, please [submit an issue on github](https://github.com/airbus/decomon/issues).
