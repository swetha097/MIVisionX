import sys

from amd.rali import readers 
from amd.rali import decoders 
from amd.rali import random
from amd.rali import reductions
from amd.rali import segmentation
from amd.rali import transforms
import inspect
from amd.rali.global_cfg import Node,add_node
import amd.rali.types as types
import rali_pybind as b


#brightness=1.0, bytes_per_sample_hint=0, image_type=0, preserve=False, seed=-1, device=None
def brightness(*inputs,brightness=1.0, bytes_per_sample_hint=0, image_type=0,
                 preserve=False, seed=-1, device= None):
    """
brightness (float, optional, default = 1.0) –

Brightness change factor. Values >= 0 are accepted. For example:

0 - black image,

1 - no change

2 - increase brightness twice

bytes_per_sample_hint (int, optional, default = 0) – Output size hint (bytes), per sample. The memory will be preallocated if it uses GPU or page-locked memory

image_type (int, optional, default = 0) – The color space of input and output image

preserve (bool, optional, default = False) – Do not remove the Op from the graph even if its outputs are unused.

seed (int, optional, default = -1) – Random seed (If not provided it will be populated based on the global seed of the pipeline)
    """
    current_node = Node()
    print(f'\n args:{inputs[0]}')
    print(inputs[0])
    print("output_image BRIGHTNESS {inputs[0].output_image}")
    print(inputs[0].output_image)
    
    kwargs_pybind = {"input_image":inputs[0].output_image, "is_output":current_node.is_output ,"alpha": None,"beta": None}
    #Node Object
    current_node.node_name = "Brightness"
    current_node.rali_c_func_call = b.Brightness
    current_node.kwargs_pybind = kwargs_pybind
    current_node.kwargs = {"brightness": brightness, "bytes_per_sample_hint": bytes_per_sample_hint, "image_type": image_type,
                           "preserve": preserve, "seed": seed, "device": device}
    current_node.has_input_image = True
    current_node.has_output_image = True

    #Connect the Prev Node(inputs[0]) < === > Current Node
    add_node(inputs[0],current_node)

    return (current_node)