from keras.layers import Dense, Input

from decomon.models import DecomonModel


def test_get_config():
    input = Input((1,))
    output = Dense(2)(Dense(3)(input))
    decomon_model = DecomonModel(input, output)
    config = decomon_model.get_config()
    print(config)
    expected_keys = [
        "layers",
        "name",
        "input_layers",
        "output_layers",
        "perturbation_domain",
        "dc_decomp",
        "method",
        "ibp",
        "affine",
        "finetune",
        "shared",
        "backward_bounds",
    ]
    for k in expected_keys:
        assert k in config
    assert len(config["layers"]) == len(decomon_model.layers)
