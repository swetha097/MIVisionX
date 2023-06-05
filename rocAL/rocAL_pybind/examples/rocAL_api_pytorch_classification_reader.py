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

import random
from amd.rocal.plugin.pytorch import ROCALClassificationIterator
from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import amd.rocal.types as types
import numpy as np
import sys
import os

def draw_patches(img, idx, layout="nchw", dtype="fp32", device="cpu"):
    #image is expected as a tensor
    import cv2
    if device == "cpu":
        image = img.detach().numpy()
    else:
        image = img.cpu().numpy()
    if layout == "nchw":
        image = image.transpose([1, 2, 0])
    if dtype == "fp16":
        image = image.astype('uint8')
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("OUTPUT_IMAGES_PYTHON/NEW_API/FILE_READER/" + str(idx)+"_"+"train"+".png", image * 255)

def main():
    if  len(sys.argv) < 3:
        print ('Please pass image_folder cpu/gpu batch_size')
        exit(0)
    try:
        path= "OUTPUT_IMAGES_PYTHON/NEW_API/FILE_READER/"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
    except OSError as error:
        print(error)
    # Create Pipeline instance
    pipe = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=args.local_rank, seed=random_seed, rocal_cpu=rocal_cpu, tensor_layout=types.NHWC if args.NHWC else types.NCHW , tensor_dtype=types.FLOAT16 if args.fp16 else types.FLOAT)
    # Set Params
    output_set = 0
    rocal_device = 'cpu' if rocal_cpu else 'gpu'
    #hardcoding decoder_device to cpu until VCN can decode all JPEGs
    decoder_device = 'cpu'
    # Use pipeline instance to make calls to reader, decoder & augmentation's
    with pipe:
        jpegs, _ = fn.readers.file(file_root=data_path, shard_id=local_rank, num_shards=world_size, random_shuffle=True)
        images = fn.decoders.image(jpegs, file_root=data_path, device=decoder_device, output_type=types.RGB, shard_id=0, num_shards=1, random_shuffle=True)
        images = fn.resize(images, device=rocal_device, resize_x=300, resize_y=300)


        if augmentation_name == "resize":
            output = fn.resize(images, device=rocal_device, resize_x=300, resize_y=300, 
                               scaling_mode=types.SCALING_MODE_NOT_SMALLER, interpolation_type=types.TRIANGULAR_INTERPOLATION)
        elif augmentation_name == "rotate":
            output = fn.rotate(images)
        elif augmentation_name == "brightness":
            output = fn.brightness(images)
        elif augmentation_name == "gamma_correction":
            output = fn.gamma_correction(images)
        elif augmentation_name == "contrast":
            output = fn.contrast(images)
        elif augmentation_name == "flip":
            output = fn.flip(images)
        elif augmentation_name == "blur":
            output = fn.blur(images)
        elif augmentation_name == "one_hot":
            _ = fn.one_hot(num_classes=2)
            output = fn.resize(images, device=rocal_device, resize_x=300, resize_y=300)
        elif augmentation_name == "hue_rotate_blend":
            images_hue = fn.hue(images)
            images_rotate = fn.rotate(images)
            output = fn.blend(images_hue, images_rotate)
        elif augmentation_name == "warp_affine":
            output = fn.warp_affine(images)
        elif augmentation_name == "fish_eye":
            output = fn.fish_eye(images)
        elif augmentation_name == "vignette":
            output = fn.vignette(images)
        elif augmentation_name == "jitter":
            output = fn.jitter(images)
        elif augmentation_name == "snp_noise":
            output = fn.snp_noise(images)
        elif augmentation_name == "snow":
            output = fn.snow(images)
        elif augmentation_name =="rain":
            output = fn.rain(images)
        elif augmentation_name == "fog":
            output = fn.fog(images)
        elif augmentation_name == "pixelate":
            output = fn.pixelate(images)
        elif augmentation_name == "exposure":
            output = fn.exposure(images)
        elif augmentation_name == "hue":
            output = fn.hue(images)
        elif augmentation_name == "saturation":
            output = fn.saturation(images)
        elif augmentation_name == "color_twist":
            output = fn.color_twist(images)
        elif augmentation_name == "crop_mirror_normalize":
            output = fn.crop_mirror_normalize(images, device="cpu",
                                              output_dtype=types.UINT8,
                                              output_layout=types.NHWC,
                                              crop=(300, 300),
                                              image_type=types.RGB,
                                              mean=[0, 0, 0],
                                              std=[1, 1, 1])
        elif augmentation_name == "nop":
            output = fn.nop(images)
        elif augmentation_name == "centre_crop":
            output = fn.centre_crop(images)
        elif augmentation_name == "color_temp":
            output = fn.color_temp(images)
        elif augmentation_name == "copy":
            output = fn.copy(images)
        elif augmentation_name == "rotate_fisheye_fog":
            output1 = fn.rotate(images)
            output2 = fn.fish_eye(output1)
            output3 = fn.fog(output2)
            pipe.set_outputs(output1, output2, output3)
            output_set = 1
        elif augmentation_name == "resize_brightness_jitter":
            output1 = fn.resize(images, resize_x=300, resize_y=300)
            output2 = fn.brightness(output1)
            output3 = fn.jitter(output2)
            pipe.set_outputs(output1, output2, output3)
            output_set = 1
        elif augmentation_name == "vignetter_blur":
            output1 = fn.vignette(images)
            output2 = fn.blur(output1)
            pipe.set_outputs(output1, output2)
            output_set = 1

        if output_set==0:
                pipe.set_outputs(output)
    # build the pipeline
    pipe.build()
    # Dataloader
    data_loader = ROCALClassificationIterator(pipe,device=device)
    cnt = 0
    for epoch in range(3):
        print("+++++++++++++++++++++++++++++EPOCH+++++++++++++++++++++++++++++++++++++",epoch)
        for i, it in enumerate(imageIteratorPipeline):
            print(it)
            print("************************************** i *************************************",i)
            for img in it[0]:
                cnt = cnt + 1
                draw_patches(img, cnt, layout="nhwc", dtype="fp16", device=_rali_cpu) 
        imageIteratorPipeline.reset()
    print("*********************************************************************")

if __name__ == '__main__':
    main()
