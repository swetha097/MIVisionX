from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import numpy as np
from amd.rocal.plugin.pytorch import ROCALClassificationIterator
import torch
torch.set_printoptions(threshold=10_000)
np.set_printoptions(threshold=1000, edgeitems=10000)

from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import amd.rocal.types as types
# import rocal_pybind.tensor
import sys
import cv2
import matplotlib.pyplot as plt
import os

def draw_patches(img, idx, device):
    #image is expected as a tensor, bboxes as numpy
    import cv2
    if device == "cpu":
            image = img.detach().numpy()
    else:
        image = img.cpu().numpy()
    # image = image.transpose([1, 2, 0])
    print(img.shape)
    print(idx)
    print(img.cpu().detach().numpy().flatten())
    print(idx)
    # exit(0)
    audio_data = img.flatten()
    label = idx.cpu().detach().numpy()
    plt.plot(audio_data)
    plt.savefig("rocal_audio_data"+str(label)+".png")
    # cv2.imwrite("OUTPUT_IMAGES_PYTHON/NEW_API/FILE_READER/" + "brightness" + "/" + str(idx)+"_"+"train"+".png", image * 255)

def main():
    if  len(sys.argv) < 3:
        print ('Please pass audio_folder file_list cpu/gpu batch_size')
        exit(0)
    try:
        path= "OUTPUT_IMAGES_PYTHON/NEW_API/FILE_READER/" + "brightness"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
    except OSError as error:
        print(error)
    data_path = sys.argv[1]
    file_list = sys.argv[2]
    if(sys.argv[3] == "cpu"):
        _rali_cpu = True
    else:
        _rali_cpu = False
    batch_size = int(sys.argv[4])
    num_threads = 1
    device_id = 0
    random_seed = random.SystemRandom().randint(0, 2**32 - 1)
    crop=300

    pipe = Pipeline(batch_size=batch_size, num_threads=num_threads,device_id=device_id, seed=random_seed, rocal_cpu=_rali_cpu)
    local_rank = 0
    world_size = 1

    print("*********************************************************************")


    audio_pipeline = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=random_seed, rocal_cpu=_rali_cpu)

    with audio_pipeline:
        audio, label = fn.readers.file(
            # **files_arg,
            file_root=data_path,
            file_list=file_list,
            shard_id=0,
            num_shards=1,)
        sample_rate = 16000
        nfft=512
        window_size=0.02
        window_stride=0.01
        nfilter=80 #nfeatures
        resample = 16000.00
        # dither = 0.001
        audio_decode = fn.decoders.audio(audio, file_root=data_path, downmix=True, shard_id=0, num_shards=1)
        distribution = uniform_distribution = fn.random.uniform(audio_decode, range=[0.85, 1.15])
        sample_rate = distribution * resample
        resample_output = fn.resample(audio_decode, resample_rate = sample_rate, resample_hint=522320*1.15, )
        # print(audio_decode)
        # audio_new = audio_decode * 1.0
        # print(audio_new)
        # dither = 0.001
        # distribution = fn.random.normal(audio_decode, mean=0.0, stddev=1.0)
        uniform_distribution = fn.random.uniform(audio_decode, range=[0.85, 1.15])
        # mul_dist1 = uniform_distribution * 16000.00
        # dither = 0.001
        # distribution = fn.random.normal(audio_decode, mean=0.0, stddev=1.0)
        # mul_dist1 = distribution * 0.0001
        # begin, length = fn.nonsilent_region(audio_decode, cutoff_db=-60)
        # trim_silence = fn.slice(
        #     audio_decode,
        #     anchor=[begin],
        #     shape=[length],
        #     normalized_anchor=False,
        #     normalized_shape=False,
        #     axes=[0]
        # )

        # mul_dist = audio_decode + distribution * 0.0001 
        # p = distribution
        # new_dist =  trim_silence
        # distributed_normalied_audio = trim_silence * new_dist
        # distribution_new = distribution * dither
        # premph_audio = fn.preemphasis_filter(audio_decode)
        # spectrogram_audio = fn.spectrogram(
        #     premph_audio,
        #     nfft=nfft,
        #     window_length=320, # Change to 320
        #     window_step= 160, # Change to 160
        #     rocal_tensor_output_type=types.FLOAT,
        # )
        # mel_filter_bank_audio = fn.mel_filter_bank(
        #     spectrogram_audio,
        #     sample_rate=sample_rate,
        #     nfilter=nfilter,
        # )
        # to_decibels_audio = fn.to_decibels(
        #     mel_filter_bank_audio,
        #     multiplier=np.log(10),
        #     reference=1.0,
        #     cutoff_db=np.log(1e-20),
        #     rocal_tensor_output_type=types.FLOAT,
        # )
        # normalize_audio = fn.normalize(to_decibels_audio, axes=[1])
        # pad_audio = fn.pad(normalize_audio, fill_value=0)
        audio_pipeline.set_outputs(resample_output)

    audio_pipeline.build()
    audioIteratorPipeline = ROCALClassificationIterator(audio_pipeline)
    cnt = 0
    for e in range(2):
        # torch.set_printoptions(threshold=5000, profile="full", edgeitems=100)
        for i , it in enumerate(audioIteratorPipeline):
            print("************************************** i *************************************",i)
            print(it)
        print("EPOCH DONE", e)
        audioIteratorPipeline.reset()



if __name__ == '__main__':
    main()
