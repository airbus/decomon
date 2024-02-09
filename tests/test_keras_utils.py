import keras
import keras.ops as K
import numpy as np
import pytest
from keras.layers import Dense, Input, Layer
from numpy.testing import assert_almost_equal

from decomon.keras_utils import (
    batch_multid_dot,
    get_weight_index_from_name,
    is_a_merge_layer,
    share_layer_all_weights,
)
from decomon.types import BackendTensor


class MyLayer(Layer):
    """Mock layer unknown from decomon."""

    ...


class MyMerge(Layer):
    """Mock merge layer unknown from decomon."""

    def _merge_function(self, inputs):
        return inputs


def test_is_merge_layer():
    layer = MyMerge()
    assert is_a_merge_layer(layer)
    layer = MyLayer()
    assert not is_a_merge_layer(layer)


def test_batch_multid_dot_symbolic_nok():
    with pytest.raises(ValueError):
        batch_multid_dot(Input((2, 5, 1)), Input((4, 8, 2, 9)), nb_merging_axes=2)


def test_batch_multid_dot_symbolic_ok():
    output = batch_multid_dot(Input((2, 5, 1)), Input((5, 1, 8, 2, 9)), nb_merging_axes=2)
    assert output.shape == (None, 2, 8, 2, 9)


def test_batch_multid_dot_nok():
    with pytest.raises(ValueError):
        batch_multid_dot(K.ones((10, 2, 5, 1)), K.ones((1, 4, 8, 2, 9)), nb_merging_axes=2)


def test_batch_multid_dot_ok():
    x = K.repeat(K.reshape(K.eye(5), (1, 5, 5, 1)), 10, axis=0)
    y = K.ones((10, 5, 1, 8, 2, 9))
    output = batch_multid_dot(x, y, nb_merging_axes=2)
    expected_shape = (10, 5, 8, 2, 9)
    assert output.shape == expected_shape
    assert K.convert_to_numpy(output == K.ones(expected_shape)).all()


def test_batch_multid_dot_missing_y_batch_nok():
    x = K.repeat(K.reshape(K.eye(5), (1, 5, 5, 1)), 10, axis=0)
    y = K.ones((10, 5, 1, 8, 2, 9))
    with pytest.raises(ValueError):
        batch_multid_dot(x, y, nb_merging_axes=2, missing_batchsize=(False, True))


def test_batch_multid_dot_missing_y_batch_ok(helpers):
    batchsize = 10
    x_shape_wo_batch = (4, 5, 2)
    y_shape_wo_batch = (5, 2, 3, 7)
    nb_merging_axes = 2
    expected_res_shape_wo_batchsize = (4, 3, 7)
    x_shape = (batchsize,) + x_shape_wo_batch

    x = K.convert_to_tensor(np.random.random(x_shape))
    y_wo_batch = K.convert_to_tensor(np.random.random(y_shape_wo_batch))
    y_w_batch = K.repeat(y_wo_batch[None], batchsize, axis=0)

    res = batch_multid_dot(x, y_wo_batch, nb_merging_axes=nb_merging_axes, missing_batchsize=(False, True))
    assert tuple(res.shape)[1:] == expected_res_shape_wo_batchsize
    res_w_all_batch = batch_multid_dot(x, y_w_batch, nb_merging_axes=nb_merging_axes)
    helpers.assert_almost_equal(res, res_w_all_batch)


def test_batch_multid_dot_missing_x_batch_ok(helpers):
    batchsize = 10
    x_shape_wo_batch = (4, 5, 2)
    y_shape_wo_batch = (5, 2, 3, 7)
    nb_merging_axes = 2
    expected_res_shape_wo_batchsize = (4, 3, 7)
    y_shape = (batchsize,) + y_shape_wo_batch

    y = K.convert_to_tensor(np.random.random(y_shape))
    x_wo_batch = K.convert_to_tensor(np.random.random(x_shape_wo_batch))
    x_w_batch = K.repeat(x_wo_batch[None], batchsize, axis=0)

    res = batch_multid_dot(x_wo_batch, y, nb_merging_axes=nb_merging_axes, missing_batchsize=(True, False))
    assert tuple(res.shape)[1:] == expected_res_shape_wo_batchsize
    res_w_all_batch = batch_multid_dot(x_w_batch, y, nb_merging_axes=nb_merging_axes)
    helpers.assert_almost_equal(res, res_w_all_batch)


@pytest.mark.parametrize("missing_batchsize", [(False, False), (True, False), (False, True)])
def test_batch_multid_dot_default_nb_merging_axes(missing_batchsize, helpers):
    batchsize = 10
    x_shape_wo_batch = (4, 5, 2)
    y_shape_wo_batch = (4, 5, 2, 3, 7)
    nb_merging_axes = len(x_shape_wo_batch)
    expected_res_shape_wo_batchsize = (3, 7)
    x_shape = (batchsize,) + x_shape_wo_batch
    y_shape = (batchsize,) + y_shape_wo_batch

    x_missing_batchsize, y_missing_batchsize = missing_batchsize
    if x_missing_batchsize:
        x = K.convert_to_tensor(np.random.random(x_shape_wo_batch))
    else:
        x = K.convert_to_tensor(np.random.random(x_shape))
    if y_missing_batchsize:
        y = K.convert_to_tensor(np.random.random(y_shape_wo_batch))
    else:
        y = K.convert_to_tensor(np.random.random(y_shape))

    res_default = batch_multid_dot(x, y, missing_batchsize=missing_batchsize)
    res = batch_multid_dot(x, y, nb_merging_axes=nb_merging_axes, missing_batchsize=missing_batchsize)
    helpers.assert_almost_equal(res, res_default)


def generate_tensor_full_n_diag(
    batchsize: int,
    diag_shape: tuple[int, ...],
    other_shape: tuple[int, ...],
    diag: bool,
    missing_batchsize: bool,
    left: bool,
) -> tuple[BackendTensor, BackendTensor]:
    batchshape = (batchsize,)
    flatten_diag_shape = (int(np.prod(diag_shape)),)
    full_shape = diag_shape + diag_shape

    if diag:
        if missing_batchsize:
            x_diag_flatten = K.convert_to_tensor(np.random.random(flatten_diag_shape), dtype=float)
            x_diag = K.reshape(x_diag_flatten, diag_shape)
            x_full = K.reshape(K.diag(x_diag_flatten), full_shape)
        else:
            x_diag_flatten = K.convert_to_tensor(np.random.random(batchshape + flatten_diag_shape), dtype=float)
            x_diag = K.reshape(x_diag_flatten, batchshape + diag_shape)
            x_full = K.concatenate(
                [K.reshape(K.diag(x_diag_flatten[i]), full_shape)[None] for i in range(batchsize)], axis=0
            )
    else:
        if left:
            x_shape = other_shape + diag_shape
        else:
            x_shape = diag_shape + other_shape
        if not missing_batchsize:
            x_shape = batchshape + x_shape
        x = K.convert_to_tensor(np.random.random(x_shape), dtype=float)
        x_full = x
        x_diag = x

    return x_full, x_diag


@pytest.mark.parametrize("missing_batchsize", [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize("diagonal", [(True, True), (True, False), (False, True)])
def test_batch_multi_dot_diag(missing_batchsize, diagonal, helpers):
    batchsize = 10
    diag_shape = (4, 5, 2)
    other_shape = (3, 7)
    nb_merging_axes = len(diag_shape)

    diag_x, diag_y = diagonal
    missing_batchsize_x, missing_batchsize_y = missing_batchsize

    x_full, x_diag = generate_tensor_full_n_diag(
        batchsize=batchsize,
        diag_shape=diag_shape,
        other_shape=other_shape,
        diag=diag_x,
        missing_batchsize=missing_batchsize_x,
        left=True,
    )
    y_full, y_diag = generate_tensor_full_n_diag(
        batchsize=batchsize,
        diag_shape=diag_shape,
        other_shape=other_shape,
        diag=diag_y,
        missing_batchsize=missing_batchsize_y,
        left=False,
    )

    res_full = batch_multid_dot(x_full, y_full, nb_merging_axes=nb_merging_axes, missing_batchsize=missing_batchsize)
    res_diag = batch_multid_dot(
        x_diag, y_diag, nb_merging_axes=nb_merging_axes, missing_batchsize=missing_batchsize, diagonal=diagonal
    )

    if diag_x and diag_y:
        # the result stayed diagonal, needs to be reworked to be compared with full result
        assert res_diag.shape == (batchsize,) + diag_shape
        res_diag = K.concatenate(
            [K.reshape(K.diag(K.ravel(res_diag[i])), diag_shape + diag_shape)[None] for i in range(len(res_diag))],
            axis=0,
        )
    elif diag_x:
        assert res_diag.shape == (batchsize,) + diag_shape + other_shape
    elif diag_y:
        assert res_diag.shape == (batchsize,) + other_shape + diag_shape

    helpers.assert_almost_equal(
        res_full,
        res_diag,
    )


def test_get_weight_index_from_name_nok_attribute():
    layer = Dense(3)
    layer(K.zeros((2, 1)))
    with pytest.raises(AttributeError):
        get_weight_index_from_name(layer=layer, weight_name="toto")


def test_get_weight_index_from_name_nok_index():
    layer = Dense(3, use_bias=False)
    layer(K.zeros((2, 1)))
    with pytest.raises(IndexError):
        get_weight_index_from_name(layer=layer, weight_name="bias")


def test_get_weight_index_from_name_ok():
    layer = Dense(3)
    layer(K.zeros((2, 1)))
    assert get_weight_index_from_name(layer=layer, weight_name="bias") in [0, 1]


def test_share_layer_all_weights_nok_original_layer_unbuilt():
    original_layer = Dense(3)
    new_layer = original_layer.__class__.from_config(original_layer.get_config())
    with pytest.raises(ValueError):
        share_layer_all_weights(original_layer=original_layer, new_layer=new_layer)


def test_share_layer_all_weights_nok_new_layer_built():
    original_layer = Dense(3)
    inp = Input((1,))
    original_layer(inp)
    new_layer = original_layer.__class__.from_config(original_layer.get_config())
    new_layer(inp)
    with pytest.raises(ValueError):
        share_layer_all_weights(original_layer=original_layer, new_layer=new_layer)


def test_share_layer_all_weights_ok():
    original_layer = Dense(3)
    inp = Input((1,))
    original_layer(inp)
    new_layer = original_layer.__class__.from_config(original_layer.get_config())
    share_layer_all_weights(original_layer=original_layer, new_layer=new_layer)

    # check same weights
    assert len(original_layer.weights) == len(new_layer.weights)
    for w in original_layer.weights:
        new_w = [ww for ww in new_layer.weights if ww.name == w.name][0]
        assert w is new_w
