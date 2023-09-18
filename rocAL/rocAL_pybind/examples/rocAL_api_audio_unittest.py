
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
from amd.rocal.plugin.pytorch import ROCALAudioIterator
import torch
from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import amd.rocal.types as types
import sys
import matplotlib.pyplot as plt
import os
import numpy as np

def draw_patches(img, idx, augmentation_name):
    #image is expected as a tensor, bboxes as numpy
    image = img.detach().numpy()
    audio_data = image.flatten()
    label = idx.cpu().detach().numpy() #TODO: Uncomment after the meta-data is enabled
    # Saving the array in a text file
    file = open("results/rocal_data_new" + str(label) + ".txt", "w+")
    content = str(audio_data)
    file.write(content)
    file.close()
    plt.plot(audio_data)
    plt.savefig("results/rocal_data_new/" + str(augmentation_name) + str(label) + ".png")
    plt.close()

def main():
    if  len(sys.argv) < 3:
        print ('Please pass audio_folder file_list cpu/gpu batch_size')
        exit(0)
    try:
        path= "OUTPUT_IMAGES_PYTHON/NEW_API/FILE_READER/" + "audio"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
    except OSError as error:
        print(error)
    data_path = sys.argv[1]
    file_list = sys.argv[2]
    device_type = sys.argv[3]
    if(device_type == "cpu"):
        _rocal_cpu = True
    else:
        _rocal_cpu = False
    batch_size = int(sys.argv[4])
    num_threads = 1
    device_id = 0
    resample = 16000.00
    random_seed = random.SystemRandom().randint(0, 2**32 - 1)
    print("*********************************************************************")
    audio_pipeline = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=random_seed, rocal_cpu=_rocal_cpu)
    with audio_pipeline:
        audio, label = fn.readers.file(
            file_root=data_path,
            file_list=file_list,
            )
        audio_decode = fn.decoders.audio(audio, file_root=data_path, file_list_path=file_list, downmix=False, shard_id=0, num_shards=1, storage_type=9, stick_to_shard=False)
        uniform_distribution_resample = fn.random.uniform(audio_decode, range=[1.15,1.15])
        resampled_rate = uniform_distribution_resample * resample
        resample_output = fn.resample(audio_decode, resample_rate=resampled_rate, resample_hint=1.15 * 38240, )
        begin, length = fn.nonsilent_region(resample_output, cutoff_db=-60)
        trim_silence = fn.slice(
            resample_output,
            anchor=[begin],
            shape=[length],
            normalized_anchor=False,
            normalized_shape=False,
            axes=[0],
            rocal_tensor_output_type = types.FLOAT)
        pre_emphasis_filter = fn.preemphasis_filter(trim_silence)

        spec = fn.spectrogram(
            pre_emphasis_filter,
            nfft=512,
            window_length=320,
            window_step=160,
            rocal_tensor_output_type = types.FLOAT)
        mel = fn.mel_filter_bank(
            spec,
            sample_rate=16000,
            nfilter=80,
        )
        to_decibels = fn.to_decibels(
            mel,
            multiplier=np.log(10),
            reference=1.0,
            cutoff_db=np.log(1e-20),
            rocal_tensor_output_type=types.FLOAT,
        )
        normalize_audio = fn.normalize(to_decibels, axes=[0])
        audio_pipeline.setOutputs(normalize_audio)
    audio_pipeline.build()
    audioIteratorPipeline = ROCALAudioIterator(audio_pipeline, auto_reset=True, device=device_type)
    for e in range(1):
        print("Epoch :: ", e)
        torch.set_printoptions(threshold=5000, profile="full", edgeitems=100)
        for i , it in enumerate(audioIteratorPipeline):
            print("************************************** i *************************************", i)
            for audio, label, roi in zip(it[0][0], it[1], it[2]):
                augmentation_name = "normalize"
                print("label: ", label)
                print("roi: ", roi)
                print("audio: ", audio)
                draw_patches(audio, label, augmentation_name)
        print("EPOCH DONE", e)
if __name__ == '__main__':
    main()
