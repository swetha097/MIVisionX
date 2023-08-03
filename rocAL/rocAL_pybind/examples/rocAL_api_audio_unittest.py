
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
from amd.rocal.plugin.pytorch import ROCALClassificationIterator
import torch
from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import amd.rocal.types as types
import sys
import matplotlib.pyplot as plt
import os

def draw_patches(img, idx, device):
    #image is expected as a tensor, bboxes as numpy
    image = img.detach().numpy()
    audio_data = image.flatten()
    # label = idx
    label = idx.cpu().detach().numpy() #TODO: Uncomment after the meta-data is enabled
    print("label: ", label)
    # Saving the array in a text file
    file = open("results/rocal_data_new"+str(label)+".txt", "w+")
    content = str(audio_data)
    file.write(content)
    file.close()
    plt.plot(audio_data)
    plt.savefig("results/rocal_data_new"+str(label)+".png")
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
    if(sys.argv[3] == "cpu"):
        _rali_cpu = True
    else:
        _rali_cpu = False
    batch_size = int(sys.argv[4])
    num_threads = 1
    device_id = 0
    random_seed = random.SystemRandom().randint(0, 2**32 - 1)
    print("*********************************************************************")
    audio_pipeline = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=random_seed, rocal_cpu=_rali_cpu)
    with audio_pipeline:
        audio, label = fn.readers.file(
            file_root=data_path,
            file_list=file_list,
            )
        audio_decode = fn.decoders.audio(audio, file_root=data_path, file_list_path=file_list, downmix=False, shard_id=0, num_shards=2, storage_type=9, stick_to_shard=False)
        # begin, length = fn.nonsilent_region(audio_decode, cutoff_db=-60)
        # trim_silence = fn.slice(
        #     audio_decode,
        #     anchor=[begin],
        #     shape=[length],
        #     normalized_anchor=False,
        #     normalized_shape=False,
        #     axes=[0],
        #     rocal_tensor_output_type = types.FLOAT)
        spec = fn.spectrogram(
            audio_decode,
            nfft=512,
            window_length=320,
            window_step=160,
            rocal_tensor_output_type = types.FLOAT)
        mel = fn.mel_filter_bank(
            spec,
            sample_rate=16000,
            nfilter=80,
        )
        audio_pipeline.setOutputs(mel)
    audio_pipeline.build()
    audioIteratorPipeline = ROCALClassificationIterator(audio_pipeline, auto_reset=True)
    cnt = 0
    for e in range(1):
        print("Epoch :: ", e)
        torch.set_printoptions(threshold=5000, profile="full", edgeitems=100)
        for i , it in enumerate(audioIteratorPipeline):
            print("************************************** i *************************************",i)
            for x in range(len(it[0])):
                for img, label in zip(it[0][x],it[1]):
                    print("label", label)
                    # print("roi", roi)
                    print("cnt", cnt)
                    print("img", img)
                    draw_patches(img, label, "cpu")
                    cnt+=1
        print("EPOCH DONE", e)
if __name__ == '__main__':
    main()

