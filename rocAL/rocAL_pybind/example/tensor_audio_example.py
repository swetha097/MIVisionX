from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
from amd.rocal.plugin.pytorch import ROCALClassificationIterator

from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import amd.rocal.types as types
# import rocal_pybind.tensor
import sys
import cv2
import os

def draw_patches(img, idx, device):
    #image is expected as a tensor, bboxes as numpy
    import cv2
    if device == "cpu":
            image = img.detach().numpy()
    else:
        image = img.cpu().numpy()
    image = image.transpose([1, 2, 0])
    print(img.shape)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("OUTPUT_IMAGES_PYTHON/NEW_API/FILE_READER/" + "brightness" + "/" + str(idx)+"_"+"train"+".png", image * 255)

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
        jpegs, labels = fn.readers.file(file_root=data_path, file_list=file_list)
        speed_perturbation_coeffs = fn.uniform(rng_range=[0.85, 1.15])
        sample_rate = 16000
        # print(speed_perturbation_coeffs)
        # exit(0)
        # audio_decode = fn.decoders.audio(jpegs, file_root=data_path, sample_rate=speed_perturbation_coeffs * sample_rate )
        audio_decode = fn.decoders.audio([], file_root=data_path, sample_rate=sample_rate )
        begin, length = fn.nonsilent_region(audio_decode) # Dont understand where to use this as input in Slice to pass as what arguments - Confused
        trim_silence = fn.slice(audio_decode, normalized_anchor=False, normalized_shape=False, axes=[0], anchor=[begin], shape=[length], fill_values=[0.3])
        # if self.dither_coeff != 0.: # Where is the normal distribution call ? , cant find in the rocal_api_paramters.h
        #     audio = audio + self.normal_distribution(audio) * self.dither_coeff
        preemph_coeff=0.97
        nfft=512
        window_size=0.02
        window_stride=0.01
        preemph_audio = fn.preemphasis_filter(trim_silence, preemph_coeff=preemph_coeff)
        spectogram = fn.spectrogram(preemph_audio, nfft=nfft, window_length=int(window_size* sample_rate), window_step= int(window_stride* sample_rate), rocal_tensor_output_type=types.FLOAT)
        nfilt=80 #nfeatures
        mel_fbank = fn.mel_filter_bank(spectogram, sample_rate=sample_rate, nfilter=nfilt, normalize=True)
        to_decibels = fn.to_decibals(mel_fbank, rocal_tensor_output_type=types.FLOAT)
        normalize = fn.normalize(to_decibels, axes=[1])
        # padded_audio = fn.pad(normalize, fill_value=0)
        #Dont see the Pad augmentation support in rocAL

        audio_pipeline.set_outputs(trim_silence)

    audio_pipeline.build()
    audioIteratorPipeline = ROCALClassificationIterator(audio_pipeline)
    cnt = 0
    for e in range(3):
        for i , it in enumerate(audioIteratorPipeline):
            print("************************************** i *************************************",i)
            print(it)
            for img in it[0]:
                print(img.shape)
                cnt = cnt + 1
                    # draw_patches(img, cnt, "cpu")
        print("EPOCH DONE", e)
        audioIteratorPipeline.reset()



if __name__ == '__main__':
    main()
