from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from math import sqrt
import torch
import random
import itertools
from amd.rali.plugin.pytorch import RALI_iterator
from amd.rali.pipeline import Pipeline
import amd.rali.fn as fn
import amd.rali.types as types
import sys
import numpy as np



def HybridPipeline(batch_size,data_dir, random_seed,num_threads,device_id,_rali_cpu ):
    #  Params for decoder
    decode_width = 500
    decode_height = 500
    shuffle = False
    shard_id = 0
    num_shards = 1
    path = data_dir

    pipe = Pipeline(batch_size=batch_size, num_threads=num_threads,device_id=device_id, seed=random_seed, rali_cpu=_rali_cpu)
    # Initializing the parameters
    # pipe.set_seed(0)
    pipe.aug_strength = 0
    #params for contrast
    pipe.min_param = pipe.create_int_param(0)
    pipe.max_param = pipe.create_int_param(255)
    #param for brightness
    pipe.alpha_param = pipe.create_float_param(0.5)
    pipe.beta_param = pipe.create_float_param(10)
    #param for colorTemp
    pipe.adjustment_param = pipe.create_int_param(0)
    #param for exposure
    pipe.shift_param = pipe.create_float_param(0.0)
    #param for SnPNoise
    pipe.sdev_param = pipe.create_float_param(0.0)
    #param for gamma
    pipe.gamma_shift_param = pipe.create_float_param(0.0)
    #param for rotate
    pipe.degree_param = pipe.create_float_param(0.0)
    #param for lens correction
    pipe.strength = pipe.create_float_param(0.0)
    pipe.zoom = pipe.create_float_param(1.0)
    #param for snow
    pipe.snow = pipe.create_float_param(0.1)
    #param for rain
    pipe.rain = pipe.create_float_param(0.1)
    pipe.rain_width = pipe.create_int_param(2)
    pipe.rain_height = pipe.create_int_param(15)
    pipe.rain_transparency = pipe.create_float_param(0.25)
    #param for blur
    pipe.blur = pipe.create_int_param(5)
    #param for jitter
    pipe.kernel_size = pipe.create_int_param(3)
    #param for hue
    pipe.hue = pipe.create_float_param(1.0)
    #param for saturation
    pipe.saturation = pipe.create_float_param(1.5)
    #param for warp affine
    pipe.affine_matrix = [0.35,0.25,0.75,1,1,1]
    #param for fog
    pipe.fog = pipe.create_float_param(0.35)
    #param for vignette
    pipe.vignette = pipe.create_float_param(50)
    #param for flip
    pipe.flip_axis = pipe.create_int_param(0)
    #param for blend
    pipe.blend = pipe.create_float_param(0.5)

    
    with pipe:
        jpegs, labels = fn.readers.file(
            file_root=path, random_shuffle=False, seed=random_seed) 
        decoded_images = fn.decoders.image(jpegs, output_type=types.RGB)
        images=fn.resize(decoded_images,resize_x=300,resize_y=300)
        outputs_1 = fn.rotate(images,angle=45)
        outputs_2 = fn.fish_eye(outputs_1)
        outputs_3 = fn.fog(outputs_2)
        outputs = [images, outputs_1, outputs_3]
        outputs_size = len(outputs)
        pipe.set_outputs(*outputs,labels)
    return pipe

def updateAugmentationParameter(pipe, augmentation):
    #values for contrast
    aug_strength = augmentation
    min = int(augmentation*100)
    max = 150 + int((1-augmentation)*100)
    pipe.update_int_param(min, pipe.min_param)
    pipe.update_int_param(max, pipe.max_param)

    #values for brightness
    alpha = augmentation*1.95
    pipe.update_float_param(alpha, pipe.alpha_param)

    #values for colorTemp
    adjustment = (augmentation*99) if ((int(augmentation*100)) % 2 == 0) else (-1*augmentation*99)
    adjustment = int(adjustment)
    pipe.update_int_param(adjustment, pipe.adjustment_param)

    #values for exposure
    shift = augmentation*0.95
    pipe.update_float_param(shift, pipe.shift_param)

    #values for SnPNoise
    sdev = augmentation*0.7
    sdev = 0.06
    pipe.update_float_param(sdev, pipe.sdev_param)

    #values for gamma
    gamma_shift = augmentation*5.0
    pipe.update_float_param(gamma_shift, pipe.gamma_shift_param)

def renew_parameters(pipe):
    pipe.curr_degree = pipe.get_float_value(pipe.degree_param)
    #values for rotation change
    pipe.degree = pipe.aug_strength * 100
    pipe.update_float_param(pipe.curr_degree+pipe.degree, pipe.degree_param)

def main():
    if  len(sys.argv) < 3:
        print ('Please pass image_folder cpu/gpu batch_size')
        exit(0)
    image_path = sys.argv[1]    
    if(sys.argv[2] == "cpu"):
        _rali_cpu = True
    else:
        _rali_cpu = False

    bs = int(sys.argv[3])
    nt = 1
    di = 0
    random_seed = random.SystemRandom().randint(0, 2**32 - 1)
    crop_size=300
    pipe = HybridPipeline(bs,image_path,random_seed,nt,di,_rali_cpu)
    data_loader = RALI_iterator(pipe)
    epochs = 1
    cnt=0
    for epoch in range(int(epochs)):
        print("EPOCH:::::",epoch)
        for i, it in enumerate(data_loader, 0):
            updateAugmentationParameter(pipe,0.5+(i/0.01))
            renew_parameters(pipe)
            cnt=cnt+1
            print("**************", i, "*******************")
            print("**************starts*******************")
            print("\nImages:\n",it[0])
            print(it[0].shape)
            print("**************ends*******************")
            print("**************", i, "*******************")
        data_loader.reset()

    print('Number of times loop iterates is:',cnt)

if __name__ == '__main__':
    main()
