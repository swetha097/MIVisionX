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


from amd.rocal import readers
from amd.rocal import decoders
from amd.rocal import random
# from amd.rocal import noise
# from amd.rocal import reductions

import amd.rocal.types as types
import rocal_pybind as b
from amd.rocal.pipeline import Pipeline


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

def resize(*inputs, bytes_per_sample_hint=0, image_type=0, interp_type=1, mag_filter= 1, max_size = [], min_filter = 1,
            minibatch_size=32, preserve=False, resize_longer=0, resize_shorter= 0, resize_depth = 0, resize_width = 0, resize_height = 0,  scaling_mode=types.SCALING_MODE_DEFAULT, interpolation_type=types.LINEAR_INTERPOLATION,
            save_attrs=False, seed=1, rocal_tensor_layout=types.NCHW, rocal_tensor_output_type=types.UINT8, temp_buffer_hint=0, device = None):
    # pybind call arguments
    kwargs_pybind = {"input_image0": inputs[0], "dest_width:" : resize_width , "dest_height": resize_height, "is_output": False, "scaling_mode": scaling_mode, "max_size": max_size, "resize_shorter": resize_shorter, 
                     "resize_longer": resize_longer, "interpolation_type": interpolation_type, "rocal_tensor_layout" : rocal_tensor_layout, "rocal_tensor_output_type" : rocal_tensor_output_type}
    resized_image = b.Resize(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (resized_image)

def crop_mirror_normalize(*inputs, bytes_per_sample_hint=0, crop=[0, 0], crop_d=0, crop_h=0, crop_pos_x=0.5, crop_pos_y=0.5, crop_pos_z=0.5,
                          crop_w=0, image_type=0, mean=[0.0], mirror=1, rocal_tensor_layout =types.NCHW, rocal_tensor_output_type = types.FLOAT, pad_output=False,
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

    if isinstance(mirror,int):
        if(mirror == 0):
            mirror = b.CreateIntParameter(0)
        else:
            mirror = b.CreateIntParameter(1)

    # pybind call arguments
    kwargs_pybind = {"input_image0": inputs[0], "crop_height":crop_height, "crop_width":crop_width, "start_x":crop_pos_x, "start_y":crop_pos_y, "mean":mean, "std_dev":std,
                     "is_output": False, "mirror": mirror, "rocal_tensor_layout" : rocal_tensor_layout, "rocal_tensor_output_type" : rocal_tensor_output_type}
    b.setSeed(seed)
    cmn = b.CropMirrorNormalize(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    Pipeline._current_pipeline._tensor_layout = rocal_tensor_layout
    Pipeline._current_pipeline._tensor_dtype = rocal_tensor_output_type
    Pipeline._current_pipeline._multiplier = list(map(lambda x: 1/x ,std))
    Pipeline._current_pipeline._offset = list(map(lambda x,y: -(x/y), mean, std))
    return (cmn)

def resize_mirror_normalize(*inputs, bytes_per_sample_hint=0, interp_type=1, mag_filter= 1, max_size = [], min_filter = 1, minibatch_size=32,
                            resize_longer=0, resize_shorter= 0, resize_depth = 0, resize_width = 0, resize_height = 0,  scaling_mode=types.SCALING_MODE_DEFAULT,
                            interpolation_type=types.LINEAR_INTERPOLATION, image_type=0, mean=[0.0], mirror=1, output_dtype=types.FLOAT, rocal_tensor_layout =types.NHWC,
                            rocal_tensor_output_type = types.FLOAT, output_layout=types.NHWC, pad_output=False, preserve=False, seed=1, std=[1.0], device=None):
    #Set Seed

    if isinstance(mirror,int):
        if(mirror == 0):
            mirror = b.CreateIntParameter(0)
        else:
            mirror = b.CreateIntParameter(1)

    # pybind call arguments
    kwargs_pybind = {"input_image0": inputs[0],  "dest_width:" : resize_width , "dest_height": resize_height, "mean":mean, "std_dev":std, "is_output": False,
                     "scaling_mode": scaling_mode, "max_size": max_size, "resize_shorter": resize_shorter, "resize_longer": resize_longer, "interpolation_type":interpolation_type, "mirror": mirror,
                     "rocal_tensor_layout" : rocal_tensor_layout, "rocal_tensor_output_type" : rocal_tensor_output_type}
    b.setSeed(seed)
    rmn = b.ResizeMirrorNormalize(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    Pipeline._current_pipeline._tensor_layout = rocal_tensor_layout
    Pipeline._current_pipeline._tensor_dtype = rocal_tensor_output_type
    Pipeline._current_pipeline._multiplier = list(map(lambda x: 1/x ,std))
    Pipeline._current_pipeline._offset = list(map(lambda x,y: -(x/y), mean, std))
    return (rmn)

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
    kwargs_pybind = {"input_image0": inputs[0], "crop_width":crop_width, "crop_height":crop_height, "is_output": False, "rocal_tensor_layout": rocal_tensor_layout, "rocal_tensor_output_type" :rocal_tensor_output_type}
    centre_cropped_image = b.CenterCropFixed(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (centre_cropped_image)

def random_bbox_crop(*inputs,all_boxes_above_threshold = True, allow_no_crop =True, aspect_ratio = None, bbox_layout = "", bytes_per_sample_hint = 0,
                crop_shape = None, input_shape = None, ltrb = True, num_attempts = 1 ,scaling =  None,  preserve = False, seed = 1, shape_layout = "",
                threshold_type ="iou", thresholds = None, total_num_attempts = 0, device = None, labels = None ):
    aspect_ratio = aspect_ratio if aspect_ratio else [1.0, 1.0]
    crop_shape = [] if crop_shape is None else crop_shape
    scaling = scaling if scaling else [1.0, 1.0]
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
    kwargs_pybind = {"all_boxes_above_threshold":all_boxes_above_threshold, "no_crop": allow_no_crop, "p_aspect_ratio":aspect_ratio, "has_shape":has_shape, "crop_width":crop_width, "crop_height":crop_height, "num_attemps":num_attempts, "p_scaling":scaling, "total_num_attempts":total_num_attempts, "seed":seed }
    random_bbox_crop = b.RandomBBoxCrop(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))

    return (random_bbox_crop,[],[],[])

def uniform(*inputs,rng_range=[-1, 1], device=None):
    output_param = b.CreateFloatUniformRand(rng_range[0], rng_range[1])
    return output_param

def color_twist(*inputs, brightness=1.0, bytes_per_sample_hint=0, contrast=1.0, hue=0.0, image_type=0,
                preserve=False, saturation=1.0, seed=-1,rocal_tensor_layout=types.NHWC, rocal_tensor_output_type=types.UINT8, device=None):
    brightness = b.CreateFloatParameter(brightness) if isinstance(
        brightness, float) else brightness
    contrast = b.CreateFloatParameter(
        contrast) if isinstance(contrast, float) else contrast
    hue = b.CreateFloatParameter(hue) if isinstance(hue, float) else hue
    saturation = b.CreateFloatParameter(saturation) if isinstance(
        saturation, float) else saturation
    # pybind call arguments
    kwargs_pybind = {"input_image0": inputs[0],  "rocal_tensor_layout": rocal_tensor_layout, "rocal_tensor_output_type" :rocal_tensor_output_type, "is_output": False,
                     "alpha": brightness, "beta": contrast, "hue": hue, "sat": saturation}
    color_twist_image = b.ColorTwist(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (color_twist_image)

def box_encoder(*inputs, anchors, bytes_per_sample_hint=0, criteria=0.5, means=None, offset=False, preserve=False, scale=1.0, seed=-1, stds=None ,device = None):
    means = means if means else [0.0, 0.0, 0.0, 0.0]
    stds = stds if stds else [1.0, 1.0, 1.0, 1.0]
    kwargs_pybind ={"anchors":anchors, "criteria":criteria, "means":means, "stds":stds, "offset":offset, "scale":scale}
    box_encoder = b.BoxEncoder(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    Pipeline._current_pipeline._BoxEncoder = True
    return (box_encoder , [])

def box_iou_matcher(*inputs, anchors, criteria=0.5, high_threshold=0.5, low_threshold=0.4, allow_low_quality_matches=True, device=None):
    kwargs_pybind ={"anchors":anchors, "criteria":criteria, "high_threshold":high_threshold, "low_threshold":low_threshold, "allow_low_quality_matches":allow_low_quality_matches}
    box_iou_matcher = b.BoxIOUMatcher(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    Pipeline._current_pipeline._BoxIOUMatcher = True
    return (box_iou_matcher , []) # check what should be the return type 
# ''''''
# def spectrogram(*inputs, bytes_per_sample_hint=[0], center_windows=True, layout=types.FT, nfft=None, power=2, preserve=False, reflect_padding=True, seed=1, window_fn=[], window_length=512, window_step=256, rocal_tensor_layout=types.NCHW, rocal_tensor_output_type=types.FLOAT) :
#     '''
#     Produces a spectrogram from a 1D signal (for example, audio).

#     Input data is expected to be one channel (shape being (nsamples,), (nsamples, 1), or (1, nsamples)) of type float32.
#     '''
#     kwargs_pybind = {"input_audio0": inputs[0], "rocal_tensor_output_type" :rocal_tensor_output_type, "is_output": False, "window_fn":window_fn,
#                      "center_windows": center_windows, "reflect_padding": reflect_padding, "layout":layout, "power": power , "nfft":nfft, "window_length": window_length, "window_step":window_step }
#     spectrogram_output = b.Spectrogram(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
#     return spectrogram_output

# def nonsilent_region(*inputs, rocal_tensor_output_type=types.FLOAT, bytes_per_sample_hint=[0], cutoff_db=-60, preserve=False, reference_power=0.0, reset_interval=8192, seed=1, window_length=2048):
#     """
#     Performs leading and trailing silence detection in an audio buffer.

#     The operator returns the beginning and length of the non-silent region by comparing the short term power calculated for window_length of the signal with a silence cut-off threshold. The signal is considered to be silent when the short_term_power_db is less than the cutoff_db. where:

#     short_term_power_db = 10 * log10( short_term_power / reference_power )

#     Unless specified otherwise, reference_power is the maximum power of the signal.
#     """
#     kwargs_pybind = {"input_audio0": inputs[0], "is_output": False, "cutoff_db": cutoff_db,
#                      "reference_power": reference_power, "reset_interval": reset_interval, "window_length":window_length }
#     print("kwargs_pybind", kwargs_pybind)
#     non_slient_region_output = b.NonSilentRegion(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
#     return non_slient_region_output

# def slice(*inputs, anchor=[], shape=[], axes=[1,0], axis_names="WH", bytes_per_sample_hint=[0], dtype=types.FLOAT, end=[], fill_values=[0.0], normalized_anchor=True, normalized_shape=True,  out_of_bounds_policy=types.ERROR, preserve=False, rel_end=[], rel_shape=[], rel_start=[], seed=1, rocal_shape=[], start=[] , rocal_tensor_output_type=types.FLOAT):
#     """
#     The slice can be specified by proving the start and end coordinates, or start coordinates and shape of the slice. Both coordinates and shapes can be provided in absolute or relative terms.

#     The slice arguments can be specified by the following named arguments:

#     start: Slice start coordinates (absolute)

#     rel_start: Slice start coordinates (relative)

#     end: Slice end coordinates (absolute)

#     rel_end: Slice end coordinates (relative)

#     shape: Slice shape (absolute)

#     rel_shape: Slice shape (relative)

#     """

#     kwargs_pybind = {"input_audio0": inputs[0], "rocal_tensor_output_type": rocal_tensor_output_type, "is_output": False, "anchor": anchor[0],
#                      "shape": shape[0], "fill_values": fill_values, "axes":axes, "normalized_anchor": normalized_anchor , "normalized_shape":normalized_shape, "out_of_bounds_policy": out_of_bounds_policy }
#     slice_output = b.audioSlice(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
#     return slice_output

# def normalize(*inputs, axes=[], axis_names="", batch=False, bytes_per_sample_hint=[0], ddof=0, epsilon=0.0, mean=0.0, preserve=False, scale=1.0, seed=1, shift=0.0, stddev=0.0 , rocal_tensor_output_type=types.FLOAT):
#     '''
#     Normalizes the input by removing the mean and dividing by the standard deviation.

#     The mean and standard deviation can be calculated internally for the specified subset of axes or can be externally provided as the mean and stddev arguments.

#     The normalization is done following the formula:

#     out = scale * (in - mean) / stddev + shift

#     The formula assumes that out and in are equally shaped tensors, but mean and stddev might be either tensors of same shape, scalars, or a mix of these.
#     '''
#     kwargs_pybind = {"input_audio0": inputs[0], "rocal_tensor_output_type": rocal_tensor_output_type, "is_output": False, "batch": batch,
#                      "axes": axes, "mean": mean, "stddev":stddev, "scale": scale , "shift":shift, "ddof": ddof , "epsilon":epsilon}
#     normalize_output = b.audioNormalize(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
#     return normalize_output

# def preemphasis_filter(*inputs, border=types.CLAMP, bytes_per_sample_hint=[0], preemph_coeff=0.97, preserve=False, seed=1,  rocal_tensor_layout=types.NCHW, rocal_tensor_output_type=types.FLOAT):
#     '''
#     Applies preemphasis filter to the input data.

#     This filter, in simple form, can be expressed by the formula:

#     Y[t] = X[t] - coeff * X[t-1]    if t > 1
#     Y[t] = X[t] - coeff * X_border  if t == 0

#     with X and Y being the input and output signal, respectively.

#     The value of X_border depends on the border argument:

#     X_border = 0                    if border_type == 'zero'
#     X_border = X[0]                 if border_type == 'clamp'
#     X_border = X[1]                 if border_type == 'reflect'
    
#     '''
#     preemph_coeff_float_param = b.CreateFloatParameter(preemph_coeff)
#     kwargs_pybind = {"input_audio0": inputs[0], "rocal_tensor_output_type" :rocal_tensor_output_type, "is_output": False,
#                      "preemph_coeff": preemph_coeff_float_param, "preemph_border_type": border}
#     preemphasis_output = b.PreEmphasisFilter(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
#     return preemphasis_output

# def to_decibels(*inputs, bytes_per_sample_hint=[0], cutoff_db=-200.0, multiplier=10.0, preserve=False, reference=0.0, seed=1 , rocal_tensor_layout=types.NCHW, rocal_tensor_output_type=types.UINT8):
#     '''
#     Converts a magnitude (real, positive) to the decibel scale.

#     Conversion is done according to the following formula:

#     min_ratio = pow(10, cutoff_db / multiplier)
#     out[i] = multiplier * log10( max(min_ratio, input[i] / reference) )
#     '''
#     kwargs_pybind = {"input_audio0": inputs[0],  "rocal_tensor_layout": rocal_tensor_layout, "rocal_tensor_output_type" :rocal_tensor_output_type, "is_output": False,
#                      "cut_off_DB": cutoff_db, "multiplier": multiplier, "magnitude_reference": reference}
#     decibel_scale = b.ToDecibels(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
#     return decibel_scale

# def tensor_mul_scalar_float(*inputs, scalar=1.0, rocal_tensor_output_type=types.FLOAT, rocal_tensor_layout=types.NCHW):

#     kwargs_pybind = {"input_image0": inputs[0], "is_output": False, "rocal_tensor_layout": rocal_tensor_layout, "rocal_tensor_output_type": rocal_tensor_output_type, "scalar": scalar}
#     tensor_mul_scalar_float = b.TensorMulScalar(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
#     return (tensor_mul_scalar_float)

# def tensor_add_tensor_float(*inputs, rocal_tensor_output_type=types.FLOAT, rocal_tensor_layout=types.NCHW):
#     kwargs_pybind = {"input_image0": inputs[0], "input_image1": inputs[1], "is_output": False, "rocal_tensor_layout": rocal_tensor_layout, "rocal_tensor_output_type": rocal_tensor_output_type}
#     tensor_add_tensor_float = b.TensorAddTensor(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
#     return (tensor_add_tensor_float)

# def resample(*inputs, resample_rate=None, rocal_tensor_output_type=types.FLOAT, resample_hint=-1):
#     kwargs_pybind = {"input_image0": inputs[0], "resample_rate": resample_rate, "rocal_tensor_output_type": rocal_tensor_output_type, "is_output": False, "resample_hint":resample_hint}
#     resample_output = b.Resample(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
#     return (resample_output)

# def mel_filter_bank(*inputs, bytes_per_sample_hint=[0], freq_high=0.0, freq_low=0.0, mel_formula=types.SLANEY, nfilter=128, normalize=True, preserve=False, sample_rate=44100.0, seed=1):
#     '''
#     Converts a spectrogram to a mel spectrogram by applying a bank of triangular filters.

#     The frequency (‘f’) dimension is selected from the input layout. In case of no layout, “f”, “ft”, or “*ft” is assumed, depending on the number of dimensions.
#     '''
#     kwargs_pybind = {"input_audio0": inputs[0], "is_output": False,
#                      "freq_high": freq_high, "freq_low": freq_low, "mel_formula":mel_formula, "nfilter": nfilter , "normalize":normalize, "sample_rate": sample_rate }
#     mel_filter_bank_output = b.MelFilterBank(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
#     return mel_filter_bank_output'
#     '''