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
    kwargs_pybind = {"input_image0":inputs[0].output_image, "is_output":current_node.is_output ,"alpha": None,"beta": None}
    #Node Object
    current_node.node_name = "Brightness"
    current_node.rali_c_func_call = b.Brightness
    current_node.kwargs_pybind = kwargs_pybind
    current_node.augmentation_node = True
    current_node.kwargs = {"brightness": brightness, "bytes_per_sample_hint": bytes_per_sample_hint, "image_type": image_type,
                           "preserve": preserve, "seed": seed, "device": device}
    current_node.has_input_image = True
    current_node.has_output_image = True

    #Connect the Prev Node(inputs[0]) < === > Current Node
    add_node(inputs[0],current_node)

    return (current_node)


def blend(*inputs,**kwargs):
    current_node = Node()
 

    kwargs_pybind = {"input_image0":inputs[0].output_image, "input_image1":inputs[1].output_image, "is_output":current_node.is_output ,"ratio":None}
    #Node Object
    current_node.node_name = "Blend"
    current_node.rali_c_func_call = b.Blend
    current_node.kwargs_pybind = kwargs_pybind
    current_node.augmentation_node = True
    current_node.has_input_image = True
    current_node.has_output_image = True

    #Connect the Prev Node(inputs[0]) < === > Current Node
    add_node(inputs[0],current_node)
    add_node(inputs[1],current_node)
    return (current_node)

def snow(*inputs, snow=0.5, device=None):
    current_node = Node()

    # pybind call arguments
    kwargs_pybind = {"input_image0": inputs[0].output_image,"is_output": current_node.is_output, "shift": None}
    current_node.node_name = "Snow"
    current_node.rali_c_func_call = b.Snow
    current_node.kwargs_pybind = kwargs_pybind
    current_node.kwargs = {"snow": snow, "device": device}  # kwargs passed to this function
    current_node.has_input_image = True
    current_node.has_output_image = True
    current_node.augmentation_node = True 

    # Connect the Prev Node < === > Current Node
    add_node(inputs[0], current_node)
    return (current_node)

def exposure(*inputs, exposure=0.5, device=None):
    current_node = Node()

    # pybind call arguments
    kwargs_pybind = {"input_image0": inputs[0].output_image,
                     "is_output": current_node.is_output, "shift": None}
    current_node.node_name = "Exposure"
    current_node.rali_c_func_call = b.Exposure
    current_node.kwargs_pybind = kwargs_pybind
    current_node.kwargs = {"exposure": exposure, "device": device}  # Ones passed to this function
    current_node.has_input_image = True
    current_node.has_output_image = True
    current_node.augmentation_node = True 

    # Connect the Prev Node < === > Current Node
    add_node(inputs[0], current_node)
    return (current_node)

def fish_eye(*inputs, device=None):
    current_node = Node()

    # pybind call arguments
    kwargs_pybind = {"input_image0": inputs[0].output_image,
                     "is_output": current_node.is_output}
    current_node.node_name = "FishEye"
    current_node.rali_c_func_call = b.FishEye
    current_node.kwargs_pybind = kwargs_pybind
    current_node.kwargs = {"device": device}  # Ones passed to this function
    current_node.has_input_image = True
    current_node.has_output_image = True
    current_node.augmentation_node = True 

    # Connect the Prev Node < === > Current Node
    add_node(inputs[0], current_node)
    return (current_node)

def fog(*inputs, fog=0.5, device=None):
    current_node = Node()

    # pybind call arguments
    kwargs_pybind = {"input_image0": inputs[0].output_image,
                     "is_output": current_node.is_output, "fog_value": None}
    current_node.node_name = "Fog"
    current_node.rali_c_func_call = b.Fog
    current_node.kwargs_pybind = kwargs_pybind
    current_node.kwargs = {"fog": fog, "device": device}  # Ones passed to this function
    current_node.has_input_image = True
    current_node.has_output_image = True
    current_node.augmentation_node = True 

    # Connect the Prev Node < === > Current Node
    add_node(inputs[0], current_node)
    return (current_node)


    # brightness=1.0, bytes_per_sample_hint=0, image_type=0, preserve=False, seed=-1, device=None
def brightness(*inputs, brightness=1.0, bytes_per_sample_hint=0, image_type=0,
               preserve=False, seed=-1, device=None):
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

    kwargs_pybind = {"input_image0": inputs[0].output_image,
                     "is_output": current_node.is_output, "alpha": None, "beta": None}
    # Node Object
    current_node.node_name = "Brightness"
    current_node.rali_c_func_call = b.Brightness
    current_node.kwargs_pybind = kwargs_pybind
    current_node.kwargs = {"brightness": brightness, "bytes_per_sample_hint": bytes_per_sample_hint, "image_type": image_type,
                           "preserve": preserve, "seed": seed, "device": device}
    current_node.has_input_image = True
    current_node.has_output_image = True
    current_node.augmentation_node = True 
    # Connect the Prev Node < === > Current Node
    add_node(inputs[0], current_node)

    return (current_node)


def blur(*inputs, blur=3, device=None):  # Init arguments
    """
    BLUR
    """
    current_node = Node()

    # pybind call arguments
    kwargs_pybind = {"input_image0": inputs[0].output_image,
                     "is_output": current_node.is_output, "sdev": None}
    current_node.node_name = "Blur"
    current_node.rali_c_func_call = b.Blur
    current_node.kwargs_pybind = kwargs_pybind
    # Ones passed to this function
    current_node.kwargs = {"blur": blur, "device": device}
    current_node.has_input_image = True
    current_node.has_output_image = True
    current_node.augmentation_node = True 

    # Connect the Prev Node < === > Current Node
    add_node(inputs[0], current_node)
    return (current_node)


def contrast(*inputs, bytes_per_sample_hint=0, contrast=1.0, image_type=0,
             preserve=False, seed=-1, device=None):
    """
    bytes_per_sample_hint (int, optional, default = 0) – Output size hint (bytes), per sample. The memory will be preallocated if it uses GPU or page-locked memory

    contrast (float, optional, default = 1.0) –

    Contrast change factor. Values >= 0 are accepted. For example:

    0 - gray image,

    1 - no change

    2 - increase contrast twice

    image_type (int, optional, default = 0) – The color space of input and output image

    preserve (bool, optional, default = False) – Do not remove the Op from the graph even if its outputs are unused.

    seed (int, optional, default = -1) – Random seed (If not provided it will be populated based on the global seed of the pipeline)

    """
    current_node = Node()

    # pybind call arguments
    kwargs_pybind = {"input_image0": inputs[0].output_image,
                     "is_output": current_node.is_output, "min": None, "max": None}
    current_node.node_name = "Contrast"
    current_node.rali_c_func_call = b.Contrast
    current_node.kwargs_pybind = kwargs_pybind
    current_node.kwargs = {"bytes_per_sample_hint": bytes_per_sample_hint, "contrast": contrast, "image_type": image_type,
                           "preserve": preserve, "seed": seed, "device": device}  # Ones passed to this function
    current_node.has_input_image = True
    current_node.has_output_image = True
    current_node.augmentation_node = True 

    # Connect the Prev Node < === > Current Node
    add_node(inputs[0], current_node)
    return (current_node)

def flip(*inputs, flip=0, device=None):
    current_node = Node()

    # pybind call arguments
    kwargs_pybind = {"input_image0": inputs[0].output_image,
                     "is_output": current_node.is_output, "flip_axis": None}
    current_node.node_name = "Flip"
    current_node.rali_c_func_call = b.Flip
    current_node.kwargs_pybind = kwargs_pybind
    current_node.kwargs = {"flip": flip, "device": device}  # Ones passed to this function
    current_node.has_input_image = True
    current_node.has_output_image = True
    current_node.augmentation_node = True 

    # Connect the Prev Node < === > Current Node
    add_node(inputs[0], current_node)
    return (current_node)

def gamma_correction(*inputs, gamma=0.5, device=None):
    current_node = Node()

    # pybind call arguments
    kwargs_pybind = {"input_image0": inputs[0].output_image,
                     "is_output": current_node.is_output, "alpha": None}
    current_node.node_name = "GammaCorrection"
    current_node.rali_c_func_call = b.GammaCorrection
    current_node.kwargs_pybind = kwargs_pybind
    current_node.kwargs = {"gamma": gamma, "device": device}  # Ones passed to this function
    current_node.has_input_image = True
    current_node.has_output_image = True
    current_node.augmentation_node = True 

    # Connect the Prev Node < === > Current Node
    add_node(inputs[0], current_node)
    return (current_node)

def hue(*inputs, bytes_per_sample_hint=0,  hue=0.0, image_type=0, 
        preserve=False, seed = -1, device = None):
    """
    bytes_per_sample_hint (int, optional, default = 0) – Output size hint (bytes), per sample. The memory will be preallocated if it uses GPU or page-locked memory

    hue (float, optional, default = 0.0) – Hue change, in degrees.

    image_type (int, optional, default = 0) – The color space of input and output image

    preserve (bool, optional, default = False) – Do not remove the Op from the graph even if its outputs are unused.

    seed (int, optional, default = -1) – Random seed (If not provided it will be populated based on the global seed of the pipeline)

    """
    current_node = Node()

    # pybind call arguments
    kwargs_pybind = {"input_image0": inputs[0].output_image,
                     "is_output": current_node.is_output, "hue": None}
    current_node.node_name = "Hue"
    current_node.rali_c_func_call = b.Hue
    current_node.kwargs_pybind = kwargs_pybind
    current_node.kwargs = {"bytes_per_sample_hint": bytes_per_sample_hint, "hue": hue, "image_type": image_type,
                           "preserve": preserve, "seed": seed, "device": device}  # Ones passed to this function
    current_node.has_input_image = True
    current_node.has_output_image = True
    current_node.augmentation_node = True 

    # Connect the Prev Node < === > Current Node
    add_node(inputs[0], current_node)
    return (current_node)

def jitter(*inputs, bytes_per_sample_hint=0, fill_value=0.0, interp_type= 0, 
        mask = 1, nDegree = 2, preserve = False, seed = -1, device = None):
    """
    bytes_per_sample_hint (int, optional, default = 0) – Output size hint (bytes), per sample. The memory will be preallocated if it uses GPU or page-locked memory

    fill_value (float, optional, default = 0.0) – Color value used for padding pixels.

    interp_type (int, optional, default = 0) – Type of interpolation used.

    mask (int, optional, default = 1) –

    Whether to apply this augmentation to the input image.

    0 - do not apply this transformation

    1 - apply this transformation

    nDegree (int, optional, default = 2) – Each pixel is moved by a random amount in range [-nDegree/2, nDegree/2].

    preserve (bool, optional, default = False) – Do not remove the Op from the graph even if its outputs are unused.

    seed (int, optional, default = -1) – Random seed (If not provided it will be populated based on the global seed of the pipeline)

    """
    current_node = Node()

    # pybind call arguments
    kwargs_pybind = {"input_image0": inputs[0].output_image,
                     "is_output": current_node.is_output, "kernel_size": None}
    current_node.node_name = "Jitter"
    current_node.rali_c_func_call = b.Jitter
    current_node.kwargs_pybind = kwargs_pybind
    current_node.kwargs = {"bytes_per_sample_hint": bytes_per_sample_hint, "fill_value": fill_value, "interp_type": interp_type,
                           "mask": mask, "nDegree": nDegree, "preserve": preserve, "seed": seed, "device": device}  # Ones passed to this function
    current_node.has_input_image = True
    current_node.has_output_image = True
    current_node.augmentation_node = True

    # Connect the Prev Node < === > Current Node
    add_node(inputs[0], current_node)
    return (current_node)

def pixelate(*inputs, device = None):
    current_node = Node()

    # pybind call arguments
    kwargs_pybind = {"input_image0": inputs[0].output_image,
                     "is_output": current_node.is_output}
    current_node.node_name = "Pixelate"
    current_node.rali_c_func_call = b.Pixelate
    current_node.kwargs_pybind = kwargs_pybind
    current_node.kwargs = {"device": device}  # Ones passed to this function
    current_node.has_input_image = True
    current_node.has_output_image = True
    current_node.augmentation_node = True

    # Connect the Prev Node < === > Current Node
    add_node(inputs[0], current_node)
    return (current_node)

def rain(*inputs, rain=0.5, device = None):
    current_node = Node()

    # pybind call arguments
    kwargs_pybind = {"input_image0": inputs[0].output_image,"is_output": current_node.is_output,
                    "rain_value": None, "rain_width": None, "rain_height": None, "rain_transparency": None}
    current_node.node_name = "Rain"
    current_node.rali_c_func_call = b.Rain
    current_node.kwargs_pybind = kwargs_pybind
    current_node.kwargs = {"rain": rain, "device": device}  # Ones passed to this function
    current_node.has_input_image = True
    current_node.has_output_image = True
    current_node.augmentation_node = True

    # Connect the Prev Node < === > Current Node
    add_node(inputs[0], current_node)
    return (current_node)

def resize(*inputs, bytes_per_sample_hint=0, image_type=0, interp_type=1, mag_filter= 1, max_size = [0.0, 0.0], min_filter = 1,
            minibatch_size=32, preserve=False, resize_longer=0.0, resize_shorter= 0.0, resize_x = 0.0, resize_y = 0.0,
            save_attrs=False, seed=1, temp_buffer_hint=0, device = None):
    """
    bytes_per_sample_hint (int, optional, default = 0) – Output size hint (bytes), per sample. The memory will be preallocated if it uses GPU or page-locked memory

    image_type (int, optional, default = 0) – The color space of input and output image.

    interp_type (int, optional, default = 1) – Type of interpolation used. Use min_filter and mag_filter to specify different filtering for downscaling and upscaling.

    mag_filter (int, optional, default = 1) – Filter used when scaling up

    max_size (float or list of float, optional, default = [0.0, 0.0]) –

    Maximum size of the longer dimension when resizing with resize_shorter. When set with resize_shorter, the shortest dimension will be resized to resize_shorter iff the longest dimension is smaller or equal to max_size. If not, the shortest dimension is resized to satisfy the constraint longest_dim == max_size. Can be also an array of size 2, where the two elements are maximum size per dimension (H, W).

    Example:

    Original image = 400x1200.

    Resized with:

        resize_shorter = 200 (max_size not set) => 200x600

        resize_shorter = 200, max_size =  400 => 132x400

        resize_shorter = 200, max_size = 1000 => 200x600

    min_filter (int, optional, default = 1) – Filter used when scaling down

    minibatch_size (int, optional, default = 32) – Maximum number of images processed in a single kernel call

    preserve (bool, optional, default = False) – Do not remove the Op from the graph even if its outputs are unused.

    resize_longer (float, optional, default = 0.0) – The length of the longer dimension of the resized image. This option is mutually exclusive with resize_shorter,`resize_x` and resize_y. The op will keep the aspect ratio of the original image.

    resize_shorter (float, optional, default = 0.0) – The length of the shorter dimension of the resized image. This option is mutually exclusive with resize_longer, resize_x and resize_y. The op will keep the aspect ratio of the original image. The longer dimension can be bounded by setting the max_size argument. See max_size argument doc for more info.

    resize_x (float, optional, default = 0.0) – The length of the X dimension of the resized image. This option is mutually exclusive with resize_shorter. If the resize_y is left at 0, then the op will keep the aspect ratio of the original image.

    resize_y (float, optional, default = 0.0) – The length of the Y dimension of the resized image. This option is mutually exclusive with resize_shorter. If the resize_x is left at 0, then the op will keep the aspect ratio of the original image.

    save_attrs (bool, optional, default = False) – Save reshape attributes for testing.

    seed (int, optional, default = -1) – Random seed (If not provided it will be populated based on the global seed of the pipeline)

    temp_buffer_hint (int, optional, default = 0) – Initial size, in bytes, of a temporary buffer for resampling. Ingored for CPU variant.
    """
    current_node = Node()

    # pybind call arguments
    kwargs_pybind = {"input_image0": inputs[0].output_image, "dest_width": resize_x, "dest_height": resize_y,
                     "is_output": current_node.is_output}
    current_node.node_name = "Resize"
    current_node.rali_c_func_call = b.Resize
    current_node.kwargs_pybind = kwargs_pybind
    current_node.kwargs = {"bytes_per_sample_hint": bytes_per_sample_hint, "image_type": image_type, "interp_type": interp_type, "mag_filter": mag_filter,
                            "max_size": max_size, "min_filter": min_filter, "minibatch_size": minibatch_size, "preserve": preserve, 
                            "resize_longer": resize_longer, "resize_shorter": resize_shorter, "resize_x": resize_x , "resize_y": resize_y,
                            "save_attrs":save_attrs, "seed": seed, "temp_buffer_hint": temp_buffer_hint, "device": device}  # Ones passed to this function
    current_node.has_input_image = True
    current_node.has_output_image = True
    current_node.augmentation_node = True

    # Connect the Prev Node < === > Current Node
    add_node(inputs[0], current_node)
    return (current_node)


def random_crop(*inputs, crop_area_factor=[0.08, 1], crop_aspect_ratio=[0.75, 1.333333], 
            crop_pox_x=0, crop_pox_y=0, device = None):
    current_node = Node()

    # pybind call arguments
    kwargs_pybind = {"input_image0": inputs[0].output_image,"is_output": current_node.is_output,
                    "crop_area_factor": None, "crop_aspect_ratio": None, "crop_pos_x": None, "crop_pos_y": None, "num_of_attempts": 20}
    current_node.node_name = "RandomCrop"
    current_node.rali_c_func_call = b.RandomCrop
    current_node.kwargs_pybind = kwargs_pybind
    current_node.kwargs = {"crop_area_factor": crop_area_factor, "crop_aspect_ratio": crop_aspect_ratio, 
                        "crop_pox_x": crop_pox_x, "crop_pox_y": crop_pox_y, "device": device}  # Ones passed to this function
    current_node.has_input_image = True
    current_node.has_output_image = True
    current_node.augmentation_node = True

    # Connect the Prev Node < === > Current Node
    add_node(inputs[0], current_node)
    return (current_node)

def rotate(*inputs, angle=0, axis=None, bytes_per_sample_hint= 0, fill_value = 0.0, interp_type = 1, keep_size = False, 
            output_dtype = -1, preserve = False, seed = -1, size = None, device = None):
    """
    angle (float) – Angle, in degrees, by which the image is rotated. For 2D data, the rotation is counter-clockwise, assuming top-left corner at (0,0) For 3D data, the angle is a positive rotation around given axis

    axis (float or list of float, optional, default = []) – 3D only: axis around which to rotate. The vector does not need to be normalized, but must have non-zero length. Reversing the vector is equivalent to changing the sign of angle.

    bytes_per_sample_hint (int, optional, default = 0) – Output size hint (bytes), per sample. The memory will be preallocated if it uses GPU or page-locked memory

    fill_value (float, optional, default = 0.0) – Value used to fill areas that are outside source image. If not specified, source coordinates are clamped and the border pixel is repeated.

    interp_type (int, optional, default = 1) – Type of interpolation used.

    keep_size (bool, optional, default = False) – If True, original canvas size is kept. If False (default) and size is not set, then the canvas size is adjusted to acommodate the rotated image with least padding possible

    output_dtype (int, optional, default = -1) – Output data type. By default, same as input type

    preserve (bool, optional, default = False) – Do not remove the Op from the graph even if its outputs are unused.

    seed (int, optional, default = -1) – Random seed (If not provided it will be populated based on the global seed of the pipeline)

    size (float or list of float, optional, default = []) – Output size, in pixels/points. Non-integer sizes are rounded to nearest integer. Channel dimension should be excluded (e.g. for RGB images specify (480,640), not (480,640,3).

    """
    current_node = Node()

    # pybind call arguments
    kwargs_pybind = {"input_image0": inputs[0].output_image,"is_output": current_node.is_output,
                    "angle": None, "dest_width": 0, "dest_height": 0}
    current_node.node_name = "Rotate"
    current_node.rali_c_func_call = b.Rotate
    current_node.kwargs_pybind = kwargs_pybind
    current_node.kwargs = {"angle": angle, "axis": axis, "bytes_per_sample_hint": bytes_per_sample_hint, "fill_value": fill_value, 
                            "interp_type": interp_type, "keep_size": keep_size, "output_dtype": output_dtype, 
                            "preserve": preserve, "seed": seed, "size": size,"device": device}  # Ones passed to this function
    current_node.has_input_image = True
    current_node.has_output_image = True
    current_node.augmentation_node = True

    # Connect the Prev Node < === > Current Node
    add_node(inputs[0], current_node)
    return (current_node)

def saturation(*inputs, bytes_per_sample_hint=0,  saturation=1.0, image_type=0, preserve=False, seed = -1, device = None):
    """
    bytes_per_sample_hint (int, optional, default = 0) – Output size hint (bytes), per sample. The memory will be preallocated if it uses GPU or page-locked memory

    image_type (int, optional, default = 0) – The color space of input and output image

    preserve (bool, optional, default = False) – Do not remove the Op from the graph even if its outputs are unused.

    saturation (float, optional, default = 1.0) –

    Saturation change factor. Values >= 0 are supported. For example:

    0 - completely desaturated image

    1 - no change to image’s saturation

    seed (int, optional, default = -1) – Random seed (If not provided it will be populated based on the global seed of the pipeline)

    """
    current_node = Node()

    # pybind call arguments
    kwargs_pybind = {"input_image0": inputs[0].output_image,
                     "is_output": current_node.is_output, "sat": None}
    current_node.node_name = "Saturation"
    current_node.rali_c_func_call = b.Saturation
    current_node.kwargs_pybind = kwargs_pybind
    current_node.kwargs = {"bytes_per_sample_hint": bytes_per_sample_hint, "saturation": saturation, "image_type": image_type,
                           "preserve": preserve, "seed": seed, "device": device}  # Ones passed to this function
    current_node.has_input_image = True
    current_node.has_output_image = True
    current_node.augmentation_node = True

    # Connect the Prev Node < === > Current Node
    add_node(inputs[0], current_node)
    return (current_node)

def ssd_random_crop(*inputs, bytes_per_sample_hint=0, num_attempts=1.0, preserve=False, seed= -1, device = None):
    """
    bytes_per_sample_hint (int, optional, default = 0) – Output size hint (bytes), per sample. The memory will be preallocated if it uses GPU or page-locked memory
    num_attempts (int, optional, default = 1) – Number of attempts.
    preserve (bool, optional, default = False) – Do not remove the Op from the graph even if its outputs are unused.
    seed (int, optional, default = -1) – Random seed (If not provided it will be populated based on the global seed of the pipeline)
    """
    current_node = Node()
    if(num_attempts == 1):
        _num_attempts = 20
    else:
        _num_attempts = num_attempts
    # pybind call arguments
    kwargs_pybind = {"input_image0": inputs[0].output_image, "is_output": current_node.is_output, "p_threshold": None, 
                    "crop_area_factor": None, "crop_aspect_ratio": None, "crop_pos_x": None, "crop_pos_y": None, "num_of_attempts": 20}
    current_node.node_name = "SSDRandomCrop"
    current_node.rali_c_func_call = b.SSDRandomCrop
    current_node.kwargs_pybind = kwargs_pybind
    current_node.kwargs = {"bytes_per_sample_hint": bytes_per_sample_hint, "num_attempts": num_attempts,
                           "preserve": preserve, "seed": seed, "device": device}  # Ones passed to this function
    current_node.has_input_image = True
    current_node.has_output_image = True
    current_node.augmentation_node = True

    # Connect the Prev Node < === > Current Node
    add_node(inputs[0], current_node)
    return (current_node)

def warp_affine(*inputs, bytes_per_sample_hint=0, fill_value=0.0, interp_type = 1, matrix = None, output_dtype = -1, preserve = False, seed = -1, size = None, device = None):
    """
    bytes_per_sample_hint (int, optional, default = 0) – Output size hint (bytes), per sample. The memory will be preallocated if it uses GPU or page-locked memory

    fill_value (float, optional, default = 0.0) – Value used to fill areas that are outside source image. If not specified, source coordinates are clamped and the border pixel is repeated.

    interp_type (int, optional, default = 1) – Type of interpolation used.

    matrix (float or list of float, optional, default = []) –

    Transform matrix (dst -> src). Given list of values (M11, M12, M13, M21, M22, M23) this operation will produce a new image using the following formula

    dst(x,y) = src(M11 * x + M12 * y + M13, M21 * x + M22 * y + M23)

    It is equivalent to OpenCV’s warpAffine operation with a flag WARP_INVERSE_MAP set.

    output_dtype (int, optional, default = -1) – Output data type. By default, same as input type

    preserve (bool, optional, default = False) – Do not remove the Op from the graph even if its outputs are unused.

    seed (int, optional, default = -1) – Random seed (If not provided it will be populated based on the global seed of the pipeline)

    size (float or list of float, optional, default = []) – Output size, in pixels/points. Non-integer sizes are rounded to nearest integer. Channel dimension should be excluded (e.g. for RGB images specify (480,640), not (480,640,3).
    """
    current_node = Node()

    # pybind call arguments
    kwargs_pybind = {"input_image0": inputs[0].output_image,"is_output": current_node.is_output,
                    "dest_width": 0, "dest_height": 0, "x0": None, "x1": None, "y0": None, "y1": None, "o0": None, "o1": None}
    current_node.node_name = "WarpAffine"
    current_node.rali_c_func_call = b.WarpAffine
    current_node.kwargs_pybind = kwargs_pybind
    current_node.kwargs = {"bytes_per_sample_hint": bytes_per_sample_hint, "fill_value": fill_value, 
                            "interp_type": interp_type, "matrix": matrix, "output_dtype": output_dtype, 
                            "preserve": preserve, "seed": seed, "size": size,"device": device}  # Ones passed to this function
    current_node.has_input_image = True
    current_node.has_output_image = True
    current_node.augmentation_node = True

    # Connect the Prev Node < === > Current Node
    add_node(inputs[0], current_node)
    return (current_node)

def vignette(*inputs, vignette=0.5, device=None):
    current_node = Node()

    # pybind call arguments
    kwargs_pybind = {"input_image0": inputs[0].output_image,"is_output": current_node.is_output, "sdev": None}
    current_node.node_name = "Vignette"
    current_node.rali_c_func_call = b.Vignette
    current_node.kwargs_pybind = kwargs_pybind
    current_node.kwargs = {"vignette": vignette, "device": device}  # Ones passed to this function
    current_node.has_input_image = True
    current_node.has_output_image = True
    current_node.augmentation_node = True

    # Connect the Prev Node < === > Current Node
    add_node(inputs[0], current_node)
    return (current_node)


def crop_mirror_normalize(*inputs, bytes_per_sample_hint=0, crop=[0.0, 0.0], crop_d=0, crop_h=0, crop_pos_x=0.5, crop_pos_y=0.5, crop_pos_z=0.5,
                          crop_w=0, image_type=0, mean=[0.0], mirror=0, output_dtype=types.FLOAT, output_layout=types.NCHW, pad_output=False,
                          preserve=False, seed=1, std=[1.0], device=None):
    current_node = Node()
    current_node.node_name = "CropMirrorNormalize"
    current_node.rali_c_func_call = b.CropMirrorNormalize
    if(len(crop) == 2):
        crop_d = crop_d
        crop_h = crop[0]
        crop_w = crop[1]
    elif(len(crop) == 3):
        crop_d = crop[0]
        crop_h = crop[1]
        crop_w = crop[2]
    else:
        crop_d = crop_d
        crop_h = crop_h
        crop_w = crop_w
    #Set Seed
    b.setSeed(seed)
    current_node.has_input_image = True
    current_node.has_output_image = True
    current_node.augmentation_node = True
    current_node.CMN = True
    if isinstance(mirror,float):
        if(mirror == 0):
            mirror = b.CreateIntParameter(0)
        else:
            mirror = b.CreateIntParameter(1)
    else:
        crop_pos_x = crop_pos_y = crop_pos_z = 1

    current_node.kwargs = {"bytes_per_sample_hint": bytes_per_sample_hint, "crop": crop, "crop_d": crop_d, "crop_h": crop_h, "crop_pos_x": crop_pos_x, "crop_pos_y": crop_pos_y, "crop_pos_z": crop_pos_z, "crop_w": crop_w, "image_type": image_type,
                           "mean": mean, "mirror": mirror, "output_dtype": output_dtype, "output_layout": output_layout, "pad_output": pad_output, "preserve": preserve, "seed": seed, "std": std, "device": device}  # Ones passed to this function
    # pybind call arguments
    kwargs_pybind = {"input_image0": inputs[0].output_image, "crop_depth":crop_d, "crop_height":crop_h, "crop_width":crop_w, "start_x":crop_pos_x, "start_y":crop_pos_y, "start_z":crop_pos_z, "mean":mean, "std_dev":std,
                     "is_output": current_node.is_output, "mirror": mirror}
    current_node.kwargs_pybind = kwargs_pybind
    # Connect the Prev Node < === > Current Node
    add_node(inputs[0], current_node)
    return (current_node)


def color_twist(*inputs, brightness=1.0, bytes_per_sample_hint=0, contrast=1.0, hue=0.0, image_type=0,
                preserve=False, saturation=1.0, seed=-1, device=None):
    current_node = Node()
    brightness = b.CreateFloatParameter(brightness) if isinstance(
        brightness, float) else brightness
    bytes_per_sample_hint = bytes_per_sample_hint
    contrast = b.CreateFloatParameter(
        contrast) if isinstance(contrast, float) else contrast
    hue = b.CreateFloatParameter(hue) if isinstance(hue, float) else hue
    saturation = b.CreateFloatParameter(saturation) if isinstance(
        saturation, float) else saturation
    current_node.node_name = "ColorTwist"
    current_node.rali_c_func_call = b.ColorTwist
    current_node.has_input_image = True
    current_node.has_output_image = True
    current_node.augmentation_node = True

    # kwargs passed to this function
    current_node.kwargs = {"brightness": brightness, "bytes_per_sample_hint": bytes_per_sample_hint, "contrast": contrast, "hue": hue, "image_type": image_type, "preserve": preserve, "saturation": saturation, "seed": seed,
                           "device": device}
    # pybind call arguments
    kwargs_pybind = {"input_image0": inputs[0].output_image, "is_output": current_node.is_output,
                     "alpha": brightness, "beta": contrast, "hue": hue, "sat": saturation}
    current_node.kwargs_pybind = kwargs_pybind
    # Connect the Prev Node < === > Current Node
    add_node(inputs[0], current_node)
    return (current_node)


def uniform(*inputs,range=[-1, 1], device=None):
    output_param = b.CreateFloatUniformRand(range[0], range[1])
    return output_param


def random_bbox_crop(*inputs,all_boxes_above_threshold = True, allow_no_crop =True, aspect_ratio = None, bbox_layout = "", bytes_per_sample_hint = 0,
                crop_shape = None, input_shape = None, ltrb = True, num_attempts = 1 ,scaling =  None,  preserve = False, seed = -1, shape_layout = "",
                threshold_type ="iou", thresholds = None, total_num_attempts = 0, device = None, labels = None ):
    current_node = Node()
    aspect_ratio = aspect_ratio if aspect_ratio else [1.0, 1.0]
    if crop_shape is None:
        crop_shape = []
    else:
        crop_shape = crop_shape
    if input_shape is None:
        input_shape = []
    else:
        input_shape = input_shape
    scaling = scaling if scaling else [1.0, 1.0]
    thresholds = thresholds if thresholds else [0.0]
    crop_begin = []
    if(len(crop_shape) == 0):
        has_shape = False
        crop_width = 0
        crop_height = 0
    else:
        has_shape = True
        crop_width = crop_shape[0]
        crop_height = crop_shape[1]
    scaling = b.CreateFloatUniformRand(scaling[0], scaling[1])
    aspect_ratio = b.CreateFloatUniformRand(aspect_ratio[0], aspect_ratio[1])

    # pybind call arguments
    kwargs_pybind = {"all_boxes_above_threshold":all_boxes_above_threshold, "no_crop": allow_no_crop, "p_aspect_ratio":aspect_ratio, "has_shape":has_shape, "crop_width":crop_width, "crop_height":crop_height, "num_attemps":num_attempts, "p_scaling":scaling, "total_num_attempts":total_num_attempts }
    current_node.node_name = "RandomBBoxCrop"
    current_node.rali_c_func_call = b.RandomBBoxCrop
    current_node.kwargs_pybind = kwargs_pybind
    current_node.kwargs = {"device": device}  # Ones passed to this function (Needs change)
    current_node.has_input_image = False
    current_node.has_output_image = False
    current_node.augmentation_node = False

    return (current_node,[],[],[])