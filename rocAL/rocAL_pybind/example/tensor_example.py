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
        images = fn.decoders.image(jpeps,file_root=data_path, output_type=types.RGB, shard_id=0, num_shards=1, random_shuffle=False)
        brightend_images = fn.brightness(images)
        brightend_images2 = fn.brightness(brightend_images)

        pipe.set_outputs(images, brightend_images, brightend_images2)

    pipe.build()
    # imageIterator = RALI_iterator(pipe)
    # Need to call pipe.run() instead of iterator now (pipe.run() name is changed to pipe.run_tensor())

    output_data_batch_1 = pipe.run()
    print("\n OUTPUT DATA!!!!: ", output_data_batch_1) # rocALTensorList 1

    cv2.imwrite("output_images0_0.jpg", cv2.cvtColor(output_data_batch_1.at(0)[0], cv2.COLOR_RGB2BGR))
    cv2.imwrite("output_images0_1.jpg", cv2.cvtColor(output_data_batch_1.at(0)[1], cv2.COLOR_RGB2BGR))
    cv2.imwrite("output_images0_2.jpg", cv2.cvtColor(output_data_batch_1.at(0)[2], cv2.COLOR_RGB2BGR))

    cv2.imwrite("output_images1_0.jpg", cv2.cvtColor(output_data_batch_1.at(1)[0], cv2.COLOR_RGB2BGR))
    cv2.imwrite("output_images1_1.jpg", cv2.cvtColor(output_data_batch_1.at(1)[1], cv2.COLOR_RGB2BGR))
    cv2.imwrite("output_images1_2.jpg", cv2.cvtColor(output_data_batch_1.at(1)[2], cv2.COLOR_RGB2BGR))

    cv2.imwrite("output_images2_0.jpg", cv2.cvtColor(output_data_batch_1.at(2)[0], cv2.COLOR_RGB2BGR))
    cv2.imwrite("output_images2_1.jpg", cv2.cvtColor(output_data_batch_1.at(2)[1], cv2.COLOR_RGB2BGR))
    cv2.imwrite("output_images2_2.jpg", cv2.cvtColor(output_data_batch_1.at(2)[2], cv2.COLOR_RGB2BGR))

    output_data_batch_2 = pipe.run()
    print("\n OUTPUT DATA BATCH 2!!!!: ", output_data_batch_2) # rocALTensorList 2
    cv2.imwrite("output_images_batch_2_0_0.jpg", cv2.cvtColor(output_data_batch_2.at(0)[0], cv2.COLOR_RGB2BGR))
    cv2.imwrite("output_images_batch_2_0_1.jpg", cv2.cvtColor(output_data_batch_2.at(0)[1], cv2.COLOR_RGB2BGR))
    cv2.imwrite("output_images_batch_2_0_2.jpg", cv2.cvtColor(output_data_batch_2.at(0)[2], cv2.COLOR_RGB2BGR))

    cv2.imwrite("output_images_batch_2_1_0.jpg", cv2.cvtColor(output_data_batch_2.at(1)[0], cv2.COLOR_RGB2BGR))
    cv2.imwrite("output_images_batch_2_1_1.jpg", cv2.cvtColor(output_data_batch_2.at(1)[1], cv2.COLOR_RGB2BGR))
    cv2.imwrite("output_images_batch_2_1_2.jpg", cv2.cvtColor(output_data_batch_2.at(1)[2], cv2.COLOR_RGB2BGR))

    cv2.imwrite("output_images_batch_2_2_0.jpg", cv2.cvtColor(output_data_batch_2.at(2)[0], cv2.COLOR_RGB2BGR))
    cv2.imwrite("output_images_batch_2_2_1.jpg", cv2.cvtColor(output_data_batch_2.at(2)[1], cv2.COLOR_RGB2BGR))
    cv2.imwrite("output_images_batch_2_2_2.jpg", cv2.cvtColor(output_data_batch_2.at(2)[2], cv2.COLOR_RGB2BGR))

    output_data_batch_3 = pipe.run()
    print("3 BATCH")
    output_data_batch_4 = pipe.run()
    print("4 BATCH")
    output_data_batch_5 = pipe.run()
    print(output_data_batch_5)

    import timeit
    start = timeit.default_timer() #Timer starts



    stop = timeit.default_timer() #Timer Stops
    print('\n Time: ', stop - start)

if __name__ == '__main__':
    main()