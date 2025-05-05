import keras
import keras.ops as K
import numpy as np
import pytest
from keras.layers import Input, Layer
from numpy.testing import assert_almost_equal

from decomon.keras_utils import add_tensors, batch_multid_dot, is_a_merge_layer
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

    if missing_batchsize:
        # reconstruct a batch axis
        x_full = K.repeat(x_full[None], batchsize, axis=0)

    return x_full, x_diag


@pytest.mark.parametrize("missing_batchsize", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("diagonal", [(True, True), (True, False), (False, True), (False, False)])
def test_batch_multi_dot_diag_missing_batchsize(missing_batchsize, diagonal, helpers):
    batchsize = 10
    diag_shape = (4, 5, 2)
    other_shape = (3, 7)
    nb_merging_axes = len(diag_shape)

    diag_x, diag_y = diagonal
    missing_batchsize_x, missing_batchsize_y = missing_batchsize

    x_full, x_simplified = generate_tensor_full_n_diag(
        batchsize=batchsize,
        diag_shape=diag_shape,
        other_shape=other_shape,
        diag=diag_x,
        missing_batchsize=missing_batchsize_x,
        left=True,
    )
    y_full, y_simplified = generate_tensor_full_n_diag(
        batchsize=batchsize,
        diag_shape=diag_shape,
        other_shape=other_shape,
        diag=diag_y,
        missing_batchsize=missing_batchsize_y,
        left=False,
    )

    res_full = batch_multid_dot(x_full, y_full, nb_merging_axes=nb_merging_axes)
    res_simplified = batch_multid_dot(
        x_simplified,
        y_simplified,
        nb_merging_axes=nb_merging_axes,
        missing_batchsize=missing_batchsize,
        diagonal=diagonal,
    )

    if missing_batchsize_x and missing_batchsize_y:
        # the result stayed w/o batch axis, needs to be added to be compared with full result
        res_simplified = K.repeat(res_simplified[None], batchsize, axis=0)

    if diag_x and diag_y:
        # the result stayed diagonal, needs to be reworked to be compared with full result
        assert res_simplified.shape == (batchsize,) + diag_shape
        res_simplified = K.concatenate(
            [
                K.reshape(K.diag(K.ravel(res_simplified[i])), diag_shape + diag_shape)[None]
                for i in range(len(res_simplified))
            ],
            axis=0,
        )
    elif diag_x:
        assert res_simplified.shape == (batchsize,) + diag_shape + other_shape
    elif diag_y:
        assert res_simplified.shape == (batchsize,) + other_shape + diag_shape

    helpers.assert_almost_equal(
        res_full,
        res_simplified,
    )


@pytest.mark.parametrize(
    "x_shape, y_shape, missing_batchsize, diagonal",
    [
        ((4, 6), (6,), (True, True), (False, False)),
        ((4, 6), (6,), (False, False), (False, False)),
        ((4, 6), (6, 6), (False, False), (True, False)),
    ],
)
def test_add_tensors_nok_incompatible_shapes(x_shape, y_shape, missing_batchsize, diagonal):
    x = K.ones(x_shape)
    y = K.ones(y_shape)
    with pytest.raises(ValueError):
        add_tensors(x, y, missing_batchsize=missing_batchsize, diagonal=diagonal)


@pytest.mark.parametrize("missing_batchsize", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("diagonal", [(True, True), (True, False), (False, True), (False, False)])
def test_add_tensors_ok(missing_batchsize, diagonal, helpers):
    batchsize = 10
    diag_shape = (4, 5, 2)
    other_shape = diag_shape

    diag_x, diag_y = diagonal
    missing_batchsize_x, missing_batchsize_y = missing_batchsize

    x_full, x_simplified = generate_tensor_full_n_diag(
        batchsize=batchsize,
        diag_shape=diag_shape,
        other_shape=other_shape,
        diag=diag_x,
        missing_batchsize=missing_batchsize_x,
        left=True,
    )
    y_full, y_simplified = generate_tensor_full_n_diag(
        batchsize=batchsize,
        diag_shape=diag_shape,
        other_shape=other_shape,
        diag=diag_y,
        missing_batchsize=missing_batchsize_y,
        left=True,
    )

    res_full = x_full + y_full
    res_simplified = add_tensors(
        x_simplified,
        y_simplified,
        missing_batchsize=missing_batchsize,
        diagonal=diagonal,
    )

    if missing_batchsize_x and missing_batchsize_y:
        # the result stayed w/o batch axis, needs to be added to be compared with full result
        res_simplified = K.repeat(res_simplified[None], batchsize, axis=0)

    if diag_x and diag_y:
        # the result stayed diagonal, needs to be reworked to be compared with full result
        assert res_simplified.shape == (batchsize,) + diag_shape
        res_simplified = K.concatenate(
            [
                K.reshape(K.diag(K.ravel(res_simplified[i])), diag_shape + diag_shape)[None]
                for i in range(len(res_simplified))
            ],
            axis=0,
        )

    helpers.assert_almost_equal(
        res_full,
        res_simplified,
    )
