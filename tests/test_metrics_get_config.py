from decomon.core import ForwardMode
from decomon.metrics.loss import DecomonLossFusion, DecomonRadiusRobust
from decomon.metrics.metric import (
    AdversarialCheck,
    AdversarialScore,
    MetricMode,
    UpperScore,
)


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
    ibp, affine, mode, perturbation_domain = True, False, MetricMode.BACKWARD, {}
    for cls in [AdversarialCheck, AdversarialScore, UpperScore]:
        layer = cls(ibp=ibp, affine=affine, mode=mode, perturbation_domain=perturbation_domain)
        config = layer.get_config()
        assert config["ibp"] == ibp
        assert config["affine"] == affine
        assert config["mode"] == mode
        assert "perturbation_domain" in config
