# Tests: how to add unit tests for your code contribution?

In this page, we focus on how to write your unit tests when contributing to the code.

For more general information
about contributing on decomon, please have a look to the [contributing](contribute) page.
If you need to know how to actually run the unit tests (with tox + pytest), have a look at Section ["Running unit tests"](contribute#running-unit-tests).

## Layers

When implementing a new decomon layer, you will need to add a dedicated unit test.
The tests on decomon layers are gathered in
- "tests/test_unary_layers.py" for layers whose corresponding keras layer is unary
  (most of them: conv2D, Dense, ...);
- "tests/test_merge_layers.py" for layers whose corresponding keras layer is merging several inputs
  (Add, Concatenate, Dot, Maximum, ...).


### Unary layers

In "tests/test_unary_layers.py" you will find the generic test `test_decomon_unary_layer` which is testing layers
on several kinds of inputs:

#### Layer inputs tested

- "simple" ones which are generated at random thanks to fixture `simple_layer_input_functions` (defined in "conftest.py")
  with given shapes:
  - "0d": (None, 1),
  - "1d": (None, 3),
  - "multid": (None, 5, 6, 2)
- "standard" ones which were already defined before the big refactoring (version <= 0.2.1), with several parameters:
  - "0d": parameter `n` => different modelled functions, shape (None, 1)
  - "1d": parameter `odd` => shape (None, 2) or (None, 3)
  - "multid": parameter `data_format` ("channels_last" or "channels_first") => shape (None, 6, 6, 2) or (None, 2, 6, 6)

#### Layers tested

This test `test_decomon_unary_layer` is parametrized by the decomon/keras layers to test as shown by these lines:
```python
@parametrize(
    "decomon_layer_class, decomon_layer_kwargs, keras_layer_class, keras_layer_kwargs",
    [
        (DecomonDense, {}, Dense, dense_keras_kwargs),
        (DecomonActivation, activation_decomon_kwargs, Activation, activation_keras_kwargs),
    ],
)
def test_decomon_unary_layer(...):
    ...
```

Each line contains a tuple giving
- decomon_layer_class: the class of the decomon layer to test
- decomon_layer_kwargs: the keyword argument to use to initialize the decomon layer. Can be an empty dictionary,
  can be used e.g. to test different slope modelling for activation layers
- keras_layer_class: the corresponding keras layer class
- keras_layer_kwargs: the keyword arguments to use to initialize the keras layer. Can be an empty dictionary.

If you want to test several parametrization of your decomon layer or of the keras layer it will have to compute the bounds,
you can either
- add a line for each parametrization
- or use a dedicated fixture for decomon_layer_kwargs and/or keras_layer_kwargs

For instance if you want to test a decomon version of `ZeroPadding2D` with several padding patterns you can do
```python
@parametrize(
    "decomon_layer_class, decomon_layer_kwargs, keras_layer_class, keras_layer_kwargs",
    [
        (DecomonDense, {}, Dense, dense_keras_kwargs),
        (DecomonActivation, activation_decomon_kwargs, Activation, activation_keras_kwargs),
        (DecomonZeroPadding2D, {}, ZeroPadding2D, dict(padding=1)),
        (DecomonZeroPadding2D, {}, ZeroPadding2D, dict(padding=(1,3))),
        (DecomonZeroPadding2D, {}, ZeroPadding2D, dict(padding=((1,3), (0,5)))),
    ],
)
def test_decomon_unary_layer(...):
    ...
```
or create the proper fixture, for instance using the utility function
[`param_fixture`](https://smarie.github.io/python-pytest-cases/pytest_goodies/#param_fixtures)
from [pytest-cases](https://smarie.github.io/python-pytest-cases)
(also very useful to make [fixture unions](https://smarie.github.io/python-pytest-cases/pytest_goodies/#fixture_union)):
```python
from pytest_cases import param_fixture

zeropadding2d_keras_kwargs = param_fixture("zeropadding2d_keras_kwargs", [
    dict(padding=1),
    dict(padding=(1,3)),
    dict(padding=((1,3), (0,5))),
])

@parametrize(
    "decomon_layer_class, decomon_layer_kwargs, keras_layer_class, keras_layer_kwargs",
    [
        (DecomonDense, {}, Dense, dense_keras_kwargs),
        (DecomonActivation, activation_decomon_kwargs, Activation, activation_keras_kwargs),
        (DecomonZeroPadding2D, {}, ZeroPadding2D, zeropadding2d_keras_kwargs),
    ],
)
def test_decomon_unary_layer(...):
    ...
```

For activation layers, a fixture is used to generate both decomon and keras kwargs, as we want to include `slope` parameter
in decomon kwargs only if the activation is "relu". Then the fixture is unpacked into 2 different fixtures
`activation_decomon_kwargs` and `activation_keras_kwargs` thanks to `unpack_fixture` (again from pytest-cases),
to be place at the proper place in the line parametrizing decomon activation layers tests. See the test file for more details.

#### Specific input shapes

Some layers do not work with any input shape, so we need to skip invalid tests. For `ZeroPadding2D` which accepts only 4D inputs, it goes:
```python
def test_decomon_unary_layer(...):
    ...

    # skip some cases where the input shape is incompatible with the layer
    if isinstance(layer, ZeroPadding2D):
        if len(keras_symbolic_layer_input.shape) != 4:
            pytest.skip("ZeroPadding2D works only with 4D inputs")

    ...

```

Worse, some layers want an input shape not present in the different inputs passed to `test_decomon_unary_layer()`.
For instance, `ZeroPadding1D` and `ZeroPadding3D` wants respectively 3D and 5D inputs.

In that case you can use the next test `test_decomon_unary_layer_specific_shapes()` (or write your own test).
The test `test_decomon_unary_layer_specific_shapes()` is also parametrized as the previous one, with one more parameter:
`input_shape_wo_batchsize` (which is the expected input shape without the batchsize). The inputs will be generated via `simple_layer_input_functions_from_input_shape_wo_batchsize()`
that can accept any kind of input_shape (contrary to "standard" inputs which have imposed shapes).

The following code will test
- `ZeroPadding1D` with symbolic input of shape (None, 2, 3)
- `ZeroPadding3D` with symbolic input of shape (None, 1, 2, 2, 3)

```python
@parametrize(
    "decomon_layer_class, decomon_layer_kwargs, keras_layer_class, keras_layer_kwargs, input_shape_wo_batchsize",
    [
        (DecomonZeroPadding1D, {}, ZeroPadding1D, dict(padding=(0, 5)), (2,3)),
        (DecomonZeroPadding3D, {}, ZeroPadding3D, dict(padding=2),  (1, 2, 2, 3)),
    ],
)
def test_decomon_unary_layer_specific_shapes(...):
    ...
```

#### Summary

To test your decomon layer for unary keras layer you have to edit "tests/test_unary_layers.py":
- import the decomon and keras layer
- add the proper line(s) in the `@parametrize` of `test_decomon_unary_layer()`
  (or `test_decomon_unary_layer_specific_shapes()` if your layer needs specific input shapes)
- skip the test according to the input shape if not all inputs are valid for the layer


### Merging layers



## Models
