
from amd.rali import readers
from amd.rali import decoders
from amd.rali import random
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

def resize(*inputs, bytes_per_sample_hint=0, image_type=0, interp_type=1, mag_filter= 1, max_size = [0.0, 0.0], min_filter = 1,
            minibatch_size=32, preserve=False, resize_longer=0.0, resize_shorter= 0.0, resize_depth = 0, resize_width = 0, resize_height = 0,
            save_attrs=False, seed=1, rocal_tensor_layout=types.NCHW, rocal_tensor_output_type=types.FLOAT, interpolation_type = 4, temp_buffer_hint=0, device = None):
    # pybind call arguments
    kwargs_pybind = {"input_image0": inputs[0], "rocal_tensor_layout" : rocal_tensor_layout, "rocal_tensor_output_type" : rocal_tensor_output_type,  "resize_depth:" : resize_depth , "resize_height": resize_height, "resize_width": resize_width, "interpolation_type" : interpolation_type,
                     "is_output": False}
    resized_image = b.Resize(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (resized_image)

def crop_mirror_normalize(*inputs, bytes_per_sample_hint=0, crop=[0, 0], crop_d=0, crop_h=0, crop_pos_x=0.5, crop_pos_y=0.5, crop_pos_z=0.5,
                          crop_w=0, image_type=0, mean=[0.0], mirror=1, output_dtype=types.FLOAT, rocal_tensor_layout =types.NCHW, rocal_tensor_output_type = types.FLOAT,output_layout=types.NCHW, pad_output=False,
                          preserve=False, seed=1, std=[1.0], device=None):

    if(len(crop) == 2):
        crop_depth = crop_d
        crop_height = crop[0]
        crop_width = crop[1]
    elif(len(crop) == 3):
        crop_depth = crop[0]
        crop_height = crop[1]
        crop_width = crop[2]
    else:
        crop_depth = crop_d
        crop_height = crop_h
        crop_width = crop_w
    #Set Seed
    b.setSeed(seed)

    if isinstance(mirror,int):
        if(mirror == 0):
            mirror = b.CreateIntParameter(0)
        else:
            mirror = b.CreateIntParameter(1)

    # pybind call arguments
    kwargs_pybind = {"input_image0": inputs[0],"rocal_tensor_layout" : rocal_tensor_layout, "rocal_tensor_output_type" : rocal_tensor_output_type, "crop_depth":crop_depth, "crop_height":crop_height, "crop_width":crop_width, "start_x":1, "start_y":1, "start_z":1, "mean":mean, "std_dev":std,
                     "is_output": False, "mirror": mirror}
    b.setSeed(seed)
    cmn = b.CropMirrorNormalize(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    Pipeline._current_pipeline._tensor_layout = output_layout
    Pipeline._current_pipeline._tensor_dtype = output_dtype
    Pipeline._current_pipeline._multiplier = list(map(lambda x: 1/x ,std))
    Pipeline._current_pipeline._offset = list(map(lambda x,y: -(x/y), mean, std))
    return (cmn)

def resize_shorter(*inputs, resize_size=0, rocal_tensor_layout=types.NHWC, rocal_tensor_output_type=types.UINT8):
    # pybind call arguments
    kwargs_pybind = {"input_image0": inputs[0], "rocal_tensor_layout": rocal_tensor_layout, "rocal_tensor_output_type" :rocal_tensor_output_type, "resize_size": resize_size,
                     "is_output": False}
    resized_image = b.ResizeShorter(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (resized_image)

def centre_crop(*inputs, bytes_per_sample_hint=0, crop=[100, 100], crop_d=1, crop_h= 0, crop_pos_x = 0.5, crop_pos_y = 0.5, crop_pos_z = 0.5,
                 crop_w=0, image_type=0, output_dtype=types.FLOAT, preserve = False, seed = 1,rocal_tensor_layout=types.NHWC, rocal_tensor_output_type=types.UINT8,  device = None):

    if(len(crop) == 2):
        crop_depth = crop_d
        crop_height = crop[0]
        crop_width = crop[1]
    elif(len(crop) == 3):
        crop_depth = crop[0]
        crop_height = crop[1]
        crop_width = crop[2]
    else:
        crop_depth = crop_d
        crop_height = crop_h
        crop_width = crop_w
    #Set Seed
    b.setSeed(seed)
    # pybind call arguments
    kwargs_pybind = {"input_image0": inputs[0], "rocal_tensor_layout": rocal_tensor_layout, "rocal_tensor_output_type" :rocal_tensor_output_type, "crop_width":crop_width, "crop_height":crop_height, "crop_depth":crop_depth, "is_output": False}
    centre_cropped_image = b.CenterCropFixed(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (centre_cropped_image)




