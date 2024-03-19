# Installation


## Prerequisites

### Python 3.9+ environment

The use of a virtual environment is recommended, and you will need to ensure that the environment use a Python version
greater than 3.9.
This can be achieved for instance either by using [conda](https://docs.conda.io/en/latest/) or by using [pyenv](https://github.com/pyenv/pyenv) (or [pyenv-win](https://github.com/pyenv-win/pyenv-win) on windows)
and [venv](https://docs.python.org/fr/3/library/venv.html) module.

The following examples show how to create a virtual environment with Python version 3.10.13 with the mentioned methods.

#### With conda (all platforms)

```shell
conda create -n do-env python=3.10.13
conda activate do-env
```

#### With pyenv + venv (Linux/MacOS)

```shell
pyenv install 3.10.13
pyenv shell 3.10.13
python -m venv do-venv
source do-venv/bin/activate
```

#### With pyenv-win + venv (Windows)

```shell
pyenv install 3.10.13
pyenv shell 3.10.13
python -m venv do-venv
do-venv\Scripts\activate
```

### Keras 3+

Decomon relies on [Keras 3](https://keras.io/keras_3/) which allows the use of several backends: Tensorflow, PyTorch, and JAX.

To use it, you need to install at least one of the backends (refer to the documentation of each backend).
You can find the version needed to be compatible with keras 3 in
the [Keras 3 compatibility matrix](https://keras.io/getting_started/#compatibility-matrix)

You choose the backend used at runtime by setting the environment variable `KERAS_BACKEND`. Read more about it in
[Keras documentation](https://keras.io/getting_started/#configuring-your-backend).


#### Warning when using backend Tensorflow 2.15

When installing Tensorflow 2.15, the version 2.15 of Keras will be installed automatically. Be sure to get back keras 3
by fully uninstalling Keras then reinstalling Keras 3:

```shell
pip uninstall keras
pip install "keras>=3"
```

The problem will not occur starting from tensorflow 2.16.


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
