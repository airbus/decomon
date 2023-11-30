"""Decomon
-------
The goal of Decomon is to provide a simple interface to the latest explanation
techniques for certified perturbation analysis
"""

from importlib.metadata import PackageNotFoundError, version

import keras

from . import layers, models
from .metrics.loss import get_adv_loss, get_lower_loss, get_model, get_upper_loss
from .models.models import get_AB as get_grid_params
from .models.models import get_AB_finetune as get_grid_slope
from .wrapper import (
    check_adv_box,
    get_adv_box,
    get_adv_noise,
    get_lower_box,
    get_lower_noise,
    get_range_box,
    get_range_noise,
    get_upper_box,
    get_upper_noise,
    refine_box,
)
from .wrapper_with_tuning import get_lower_box_tuning, get_upper_box_tuning

try:
    __version__ = version("decomon")
except PackageNotFoundError:
    # package is not installed
    pass


# Check keras version
if not hasattr(keras, "__version__") or not keras.__version__.startswith("3"):
    raise RuntimeError(
        "Decomon is now compatible only with keras >= 3.0. "
        "Please check your keras version. "
        "You might have a keras 2 installed forcefully by tensorflow. "
        "In that case, you just need to reinstall keras 3 after installing tensorflow."
    )
