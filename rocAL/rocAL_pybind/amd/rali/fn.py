
from amd.rali import readers
from amd.rali import decoders
# from amd.rali import random
# from amd.rali import noise
# from amd.rali import reductions

import amd.rali.types as types
import rali_pybind as b
from amd.rali.pipeline import Pipeline


def brightness(*inputs, brightness=1.0, bytes_per_sample_hint=0, image_type=0,
               preserve=False, seed=-1, device=None):
    kwargs_pybind = {"input_image0": inputs[0], "is_output": False, "alpha": None, "beta": None}
    brightness_image = b.Brightness(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (brightness_image)

def brightness_fixed(*inputs, alpha=None, beta=None, seed=-1, device=None):
    alpha = b.CreateFloatParameter(alpha) if isinstance(alpha, float) else alpha
    beta = b.CreateFloatParameter(beta) if isinstance(beta, float) else beta
    kwargs_pybind = {"input_image0": inputs[0], "is_output": False, "alpha": alpha, "beta": beta}
    brightness_image = b.Brightness(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (brightness_image)





