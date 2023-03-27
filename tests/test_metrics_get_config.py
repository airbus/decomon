from decomon.layers.core import ForwardMode
from decomon.metrics.loss import DecomonLossFusion, DecomonRadiusRobust
from decomon.metrics.metric import AdversarialCheck, AdversarialScore, UpperScore


def test_decomon_loss():
    backward = True
    mode = ForwardMode.AFFINE

    layer = DecomonLossFusion(mode=mode, backward=backward)
    config = layer.get_config()
    assert config["backward"] == backward
    assert config["mode"] == mode

    layer = DecomonRadiusRobust(backward=backward, mode=mode)
    config = layer.get_config()
    assert config["backward"] == backward
    assert config["mode"] == mode


def test_metric():
    ibp, forward, mode, convex_domain = True, False, ForwardMode.AFFINE, {}
    for cls in [AdversarialCheck, AdversarialScore, UpperScore]:
        layer = cls(ibp=ibp, forward=forward, mode=mode, convex_domain=convex_domain)
        config = layer.get_config()
        assert config["ibp"] == ibp
        assert config["forward"] == forward
        assert config["mode"] == mode
        assert "convex_domain" in config
