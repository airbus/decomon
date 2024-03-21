from enum import Enum


class Slope(str, Enum):
    V_SLOPE = "volume-slope"
    A_SLOPE = "adaptative-slope"
    S_SLOPE = "same-slope"
    Z_SLOPE = "zero-lb"
    O_SLOPE = "one-lb"


class Propagation(str, Enum):
    """Propagation direction."""

    FORWARD = "forward"
    BACKWARD = "backward"


class ConvertMethod(str, Enum):
    CROWN = "crown"
    """Crown fully recursive: backward propagation using crown oracle.

    (spawning subcrowns for each non-linear layer)

    """
    CROWN_FORWARD_IBP = "crown-forward-ibp"
    """Crown + forward ibp: backward propagation using a forward-ibp oracle."""
    CROWN_FORWARD_AFFINE = "crown-forward-affine"
    """Crown + forward ibp: backward propagation using a forward-affine oracle."""
    CROWN_FORWARD_HYBRID = "crown-forward-hybrid"
    """Crown + forward ibp: backward propagation using a forward-hybrid oracle."""
    FORWARD_IBP = "forward-ibp"
    """Forward propagation of constant bounds."""
    FORWARD_AFFINE = "forward-affine"
    """Forward propagation of affine bounds."""
    FORWARD_HYBRID = "forward-hybrid"
    """Forward propagation of constant+affine bounds.

    After each layer, the tightest constant bounds is keep between the ibp one
    and the affine one combined with perturbation domain input.

    """
