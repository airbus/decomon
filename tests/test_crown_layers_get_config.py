from decomon.core import BoxDomain, ForwardMode
from decomon.models.crown import Convert2BackwardMode, Fuse, MergeWithPrevious


def test_crown_layers():
    mode = ForwardMode.AFFINE
    perturbation_domain = BoxDomain()
    input_shape_layer = (1, 2, 4)
    backward_shape_layer = (2, 5, 10)

    layer = Fuse(mode=mode)
    config = layer.get_config()
    assert config["mode"] == mode

    layer = Convert2BackwardMode(mode=mode, perturbation_domain=perturbation_domain)
    config = layer.get_config()
    assert config["mode"] == mode
    assert "perturbation_domain" in config

    layer = MergeWithPrevious(input_shape_layer=input_shape_layer, backward_shape_layer=backward_shape_layer)
    config = layer.get_config()
    assert config["input_shape_layer"] == input_shape_layer
    assert config["backward_shape_layer"] == backward_shape_layer
