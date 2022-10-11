from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
from amd.rali.plugin.pytorch import RALIClassificationIterator

from amd.rali.pipeline import Pipeline
import amd.rali.fn as fn
import amd.rali.types as types
# import rali_pybind.tensor
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
        audio_decode = fn.decoders.audio(jpegs, file_root=data_path)
        to_decibels = fn.to_decibals(audio_decode, rocal_tensor_output_type=types.FLOAT)
        audio_pipeline.set_outputs(to_decibels)

    audio_pipeline.build()
    audioIteratorPipeline = RALIClassificationIterator(audio_pipeline)
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
