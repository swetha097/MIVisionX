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
        brightend_images2 = fn.brightness(brightend_images)

        pipe.set_outputs(images, brightend_images, brightend_images2)

    pipe.build()
    # imageIterator = RALI_iterator(pipe)
    # Need to call pipe.run() instead of iterator now (pipe.run() name is changed to pipe.run_tensor())
    pipe.run()
    output_data_batch_1 = pipe.run_tensor()
    print("\n OUTPUT DATA!!!!: ", output_data_batch_1) # rocALTensorList

    # print("\n OUTPUT DATA 0 ::", output_data_batch_1.at(0)) # Decoder Node rocALTensor
    # print("\n OUTPUT DATA 1 ::", output_data_batch_1.at(1)) # Brightness Node  1 rocALTensor
    # print("\n OUTPUT DATA 2 ::", output_data_batch_1.at(2)) # Brightness Node  2 rocALTensor

    # print("\n OUTPUT DATA 0 ::", output_data_batch_1.at(0).shape) # Decoder Node rocALTensor
    # print("\n OUTPUT DATA 1 ::", output_data_batch_1.at(1).shape) # Brightness Node  1 rocALTensor
    # print("\n OUTPUT DATA 2 ::", output_data_batch_1.at(2).shape) # Brightness Node  2 rocALTensor


    cv2.imwrite("output_images0_0.jpg", cv2.cvtColor(output_data_batch_1.at(0)[0], cv2.COLOR_RGB2BGR))
    cv2.imwrite("output_images0_1.jpg", cv2.cvtColor(output_data_batch_1.at(0)[1], cv2.COLOR_RGB2BGR))
    cv2.imwrite("output_images0_2.jpg", cv2.cvtColor(output_data_batch_1.at(0)[2], cv2.COLOR_RGB2BGR))

    cv2.imwrite("output_images1_0.jpg", cv2.cvtColor(output_data_batch_1.at(1)[0], cv2.COLOR_RGB2BGR))
    cv2.imwrite("output_images1_1.jpg", cv2.cvtColor(output_data_batch_1.at(1)[1], cv2.COLOR_RGB2BGR))
    cv2.imwrite("output_images1_2.jpg", cv2.cvtColor(output_data_batch_1.at(1)[2], cv2.COLOR_RGB2BGR))

    cv2.imwrite("output_images0.jpg", cv2.cvtColor(output_data_batch_1.at(2)[0], cv2.COLOR_RGB2BGR))
    cv2.imwrite("output_images1.jpg", cv2.cvtColor(output_data_batch_1.at(2)[1], cv2.COLOR_RGB2BGR))
    cv2.imwrite("output_images2.jpg", cv2.cvtColor(output_data_batch_1.at(2)[2], cv2.COLOR_RGB2BGR))

    if pipe.getRemainingImages() >0:
        print("YES")
        pipe.run()

        output_data_batch_2 = pipe.run_tensor()
        print("\n OUTPUT DATA BATCH 2!!!!: ", output_data_batch_2) # rocALTensorList
        cv2.imwrite("output_images_batch_2_0_0.jpg", cv2.cvtColor(output_data_batch_2.at(0)[0], cv2.COLOR_RGB2BGR))
        cv2.imwrite("output_images_batch_2_0_1.jpg", cv2.cvtColor(output_data_batch_2.at(0)[1], cv2.COLOR_RGB2BGR))
        cv2.imwrite("output_images_batch_2_0_2.jpg", cv2.cvtColor(output_data_batch_2.at(0)[2], cv2.COLOR_RGB2BGR))

        cv2.imwrite("output_images_batch_2_1_0.jpg", cv2.cvtColor(output_data_batch_2.at(1)[0], cv2.COLOR_RGB2BGR))
        cv2.imwrite("output_images_batch_2_1_1.jpg", cv2.cvtColor(output_data_batch_2.at(1)[1], cv2.COLOR_RGB2BGR))
        cv2.imwrite("output_images_batch_2_1_2.jpg", cv2.cvtColor(output_data_batch_2.at(1)[2], cv2.COLOR_RGB2BGR))

        cv2.imwrite("output_images_batch_2_0.jpg", cv2.cvtColor(output_data_batch_2.at(2)[0], cv2.COLOR_RGB2BGR))
        cv2.imwrite("output_images_batch_2_1.jpg", cv2.cvtColor(output_data_batch_2.at(2)[1], cv2.COLOR_RGB2BGR))
        cv2.imwrite("output_images_batch_2_2.jpg", cv2.cvtColor(output_data_batch_2.at(2)[2], cv2.COLOR_RGB2BGR))


    # exit(0)
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