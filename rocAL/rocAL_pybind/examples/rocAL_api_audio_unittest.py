import random
import numpy as np
from amd.rocal.plugin.pytorch import ROCALAudioIterator
import torch
np.set_printoptions(threshold=1000, edgeitems=10000)
from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import sys
import matplotlib.pyplot as plt
import os

def plot_1d_audio(img, idx):
    #image is expected as a tensor, bboxes as numpy
    image = img.detach().numpy()
    audio_data = image.flatten()
    label = idx.cpu().detach().numpy()
    print("label: ", label)
    # Saving the array in a text file
    file = open("OUTPUTS_PYTHON/AUDIO/" + str(label) + ".txt", "w+")
    content = str(audio_data)
    file.write(content)
    file.close()
    plt.plot(audio_data)
    plt.savefig("OUTPUTS_PYTHON/AUDIO/" + str(label) + ".png")
    plt.close()

def main():
    if  len(sys.argv) < 3:
        print ('Please pass audio_folder file_list cpu/gpu batch_size')
        exit(0)
    try:
        path= "OUTPUTS_PYTHON/AUDIO/"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
    except OSError as error:
        print(error)
    data_path = sys.argv[1]
    file_list = sys.argv[2]
    if(sys.argv[3] == "cpu"):
        _rocal_cpu = True
    else:
        _rocal_cpu = False
    batch_size = int(sys.argv[4])
    num_threads = 1
    device_id = 0
    random_seed = random.SystemRandom().randint(0, 2**32 - 1)
    print("*********************************************************************")
    audio_pipeline = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=random_seed, rocal_cpu=_rocal_cpu)
    with audio_pipeline:
        audio, label = fn.readers.file(
            file_root=data_path,
            file_list=file_list,
            )
        audio_decode = fn.decoders.audio(audio, file_root=data_path, file_list_path=file_list, downmix=False, shard_id=0, num_shards=2, storage_type=9, stick_to_shard=False)
        pre_emphasis_filter = fn.preemphasis_filter(audio_decode)
        audio_pipeline.set_outputs(pre_emphasis_filter)
    audio_pipeline.build()
    audioIteratorPipeline = ROCALAudioIterator(audio_pipeline)
    cnt = 0
    for epoch in range(1):
        print("Epoch :: ", epoch)
        torch.set_printoptions(threshold=5000, profile="full", edgeitems=100)
        for i , it in enumerate(audioIteratorPipeline):
            print("************************************** i *************************************",i)
            for x in range(len(it[0])):
                for audio_data, label, roi in zip(it[0][x], it[1], it[2]):
                    print("label", label)
                    print("cnt", cnt)
                    print("roi", roi)
                    print("audio_data", audio_data)
                    plot_1d_audio(audio_data, label)
                    cnt+=1
        print("EPOCH DONE", epoch)
if __name__ == '__main__':
    main()

