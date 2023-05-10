# Copyright (c) 2018 - 2023 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

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
