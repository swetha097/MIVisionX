from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
# from amd.rali.plugin.pytorch import RALI_iterator

from amd.rali.pipeline import Pipeline
import amd.rali.fn as fn
import amd.rali.types as types
# import rali_pybind.tensor
import sys
import cv2
import os

def main():
    if  len(sys.argv) < 3:
        print ('Please pass image_folder cpu/gpu batch_size')
        exit(0)
    data_path = sys.argv[1]
    if(sys.argv[2] == "cpu"):
        _rali_cpu = True
    else:
        _rali_cpu = False
    batch_size = int(sys.argv[3])
    num_threads = 1
    device_id = 0
    random_seed = random.SystemRandom().randint(0, 2**32 - 1)
    crop=300

    pipe = Pipeline(batch_size=batch_size, num_threads=num_threads,device_id=device_id, seed=random_seed, rocal_cpu=_rali_cpu)
    local_rank = 0
    world_size = 1
    rali_cpu= True
    rali_device = 'cpu' if rali_cpu else 'gpu'
    decoder_device = 'cpu' if rali_cpu else 'mixed'
    device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
    host_memory_padding = 140544512 if decoder_device == 'mixed' else 0

    with pipe:
        # jpegs, labels = fn.readers.file(file_root=data_path)
        jpeps = [1]
        images = fn.decoders.image(jpeps,file_root=data_path, output_type=types.RGB, shard_id=0, num_shards=1, random_shuffle=True)
        brightend_images = fn.brightness(images)
        pipe.set_outputs( brightend_images)

    pipe.build()
    # imageIterator = RALI_iterator(pipe)
    output_data = pipe.run_tensor()
    print("OUTPUT DATA!!!!: ", output_data)
    # exit(0)
    #Need to call pipe.run() instead of iterator now
    # epochs = 2
    import timeit
    start = timeit.default_timer() #Timer starts

    # for epoch in range(int(epochs)):
    #     try:
    #         path= "OUTPUT_IMAGES_PYTHON/NEW_API/FILE_READER/"+"epoch"+str(epoch)+"/"
    #         os.makedirs(path, exist_ok=True)
    #     except OSError as error:
    #         print(error)
    #     print("EPOCH:::::",epoch)
    #     for i, (image_batch, image_tensor) in enumerate(imageIterator, 0):
    #             cv2.imwrite(path+"output_images_"+str(i)+".jpg", cv2.cvtColor(image_batch, cv2.COLOR_RGB2BGR))
    #     imageIterator.reset()

    stop = timeit.default_timer() #Timer Stops
    print('\n Time: ', stop - start)

if __name__ == '__main__':
    main()