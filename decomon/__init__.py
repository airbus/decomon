"""
Decomon
-------
The goal of Decomon is to provide a simple interface to the latest explanation
techniques for certified perturbation analysis
"""
from __future__ import absolute_import

import sys

from . import layers
from . import models
from .utils import get_AB as get_grid_params
from .utils import get_AB_finetune as get_grid_slope

from .wrapper import (
    get_adv_box,
    check_adv_box,
    get_upper_box,
    get_lower_box,
    get_range_box,
    get_lower_noise,
    get_range_noise,
    get_upper_noise,
    refine_box,
    get_adv_noise
)

from .wrapper_with_tuning import get_upper_box_tuning, get_lower_box_tuning
from .metrics.loss import get_model, get_upper_loss, get_lower_loss, get_adv_loss


if sys.version_info >= (3, 8):
    from importlib.metadata import (  # pylint: disable=no-name-in-module
        PackageNotFoundError,
        version,
    )
else:
    from importlib_metadata import PackageNotFoundError, version


try:
    __version__ = version("decomon")
except PackageNotFoundError:
    # package is not installed
    pass
