import rocal_pybind as b
import amd.rocal.types as types
from amd.rocal.pipeline import Pipeline

def coin_flip(*inputs,probability=0.5, device=None):
    values = [0, 1]
    frequencies = [1-probability, probability]
    output_array = b.CreateIntRand(values, frequencies)
    return output_array


def normal(*inputs, mean=0.0, stddev=1.0, dtype=types.FLOAT):
    kwargs_pybind = { "inputs":inputs[0],"is_output": False, "mean": mean, "stddev": stddev }
    output_normal_distribution = b.NormalDistribution(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    print("output_normal_distribution",output_normal_distribution)
    return (output_normal_distribution)

def uniform(*inputs, range=[-1.0, 1.0], dtype=types.FLOAT):
    kwargs_pybind = { "inputs":inputs[0],"is_output": False, "range":range }
    output_uniform_distribution = b.UniformDistribution(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    print("output_uniform_distribution",output_uniform_distribution)
    return (output_uniform_distribution)
