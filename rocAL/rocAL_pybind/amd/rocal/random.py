import rocal_pybind as b
import amd.rocal.types as types
from amd.rocal.pipeline import Pipeline

def coin_flip(*inputs,probability=0.5, device=None):
    values = [0, 1]
    frequencies = [1-probability, probability]
    output_array = b.CreateIntRand(values, frequencies)
    return output_array

def uniform(*inputs,range=[-1, 1], device=None):
    output_param = b.CreateFloatUniformRand(range[0], range[1])
    return output_param

def normal(*inputs, mean=0.0, stddev=1.0, dtype=types.FLOAT):
    kwargs_pybind = {"inputs":inputs[0],"is_output": False, "mean": mean, "stddev": stddev}
    output_normal_distribution = b.NormalDistribution(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    print("output_normal_distribution",output_normal_distribution)
    return (output_normal_distribution)