import importlib

import keras
import pytest

import decomon


def test_ok_with_keras3():
    importlib.reload(decomon)


def test_nok_with_keras2():
    keras.__version__ = "2.15"
    with pytest.raises(RuntimeError):
        importlib.reload(decomon)
