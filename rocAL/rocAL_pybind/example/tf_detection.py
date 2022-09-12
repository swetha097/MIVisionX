from __future__ import absolute_import

from __future__ import division
from __future__ import print_function
import random

from amd.rali.pipeline import Pipeline
import amd.rali.fn as fn
import amd.rali.types as types
# import rali_pybind.tensor
import sys
import cv2
import os
import tensorflow as tf
from amd.rali.plugin.tf import RALIIterator, RALI_iterator
from amd.rali.pipeline import Pipeline
import amd.rali.types as types
import amd.rali.fn as fn


import tensorflow as tf
import numpy as np
from PIL import Image as im




def draw_patches(img, idx, device):
    print("DRAW PATCHES!!")
    #image is expected as a tensor, bboxes as numpy
    import cv2 
    print("IN DRAW_PATCH  ",img.shape)
    # print(type(img))
    
    # image = img.transpose([1, 2, 0])
    # image = img.transpose([0,2,3,1])
    
    
    image =img
    print(image)
    # print(image.dtype)
    print(image.shape)
    for i in range (image.shape[0]):
        image1 = cv2.cvtColor(image[i], cv2.COLOR_RGB2BGR)
        cv2.imwrite("OUTPUT_IMAGES_PYTHON/NEW_API/FILE_READER/" + "brightness" + "/" + str(idx)+"_"+"train"+".png", image1 )
def main():
    if  len(sys.argv) < 1:
        print ('Please pass image_folder cpu/gpu batch_size')
        exit(0)
    try:
        path= "OUTPUT_IMAGES_PYTHON/NEW_API/FILE_READER/" + "brightness"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
    except OSError as error:
        print(error)
    data_path = sys.argv[1]
    if(sys.argv[2] == "cpu"):
        _rali_cpu = True
    else:
        _rali_cpu = False
    batch_size = int(sys.argv[3])
    TFRecordReaderType = 1
    featureKeyMap = {
        'image/encoded': 'image/encoded',
        'image/class/label': 'image/object/class/label',
        'image/class/text': 'image/object/class/text',
        'image/object/bbox/xmin': 'image/object/bbox/xmin',
        'image/object/bbox/ymin': 'image/object/bbox/ymin',
        'image/object/bbox/xmax': 'image/object/bbox/xmax',
        'image/object/bbox/ymax': 'image/object/bbox/ymax',
        'image/filename': 'image/filename'
    }
    num_threads = 1
    device_id = 0
    random_seed = random.SystemRandom().randint(0, 2**32 - 1)
    crop=300

    local_rank = 0
    world_size = 1
    rali_cpu= True
    rali_device = 'cpu' if rali_cpu else 'gpu'
    decoder_device = 'cpu' if rali_cpu else 'mixed'
    device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
    host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
    
    pipe = Pipeline(batch_size=batch_size, num_threads=num_threads,device_id=device_id, seed=2, rocal_cpu=_rali_cpu)

    with pipe:
        inputs = fn.readers.tfrecord(path=data_path, index_path = "", reader_type=TFRecordReaderType, user_feature_key_map=featureKeyMap,
            features={
                    'image/encoded': tf.io.FixedLenFeature((), tf.string, ""),
                    'image/class/label': tf.io.FixedLenFeature([1], tf.int64,  -1),
                    'image/class/text': tf.io.FixedLenFeature([], tf.string, ''),
                    'image/object/bbox/xmin': tf.io.VarLenFeature(dtype=tf.float32),
                    'image/object/bbox/ymin': tf.io.VarLenFeature(dtype=tf.float32),
                    'image/object/bbox/xmax': tf.io.VarLenFeature(dtype=tf.float32),
                    'image/object/bbox/ymax': tf.io.VarLenFeature(dtype=tf.float32),
                    'image/filename': tf.io.FixedLenFeature((), tf.string, "")
                    }
        )
        jpegs = inputs["image/encoded"]
        images = fn.decoders.image(jpegs, user_feature_key_map=featureKeyMap, output_type=types.RGB, path=data_path)
        resized = fn.resize(images, resize_width=300, resize_height=300)
        bright = fn .blur(images)
        pipe.set_outputs(images)
        # pipe.set_outputs(images)

        # Build the pipeline
        pipe.build()
        # Dataloader
        imageIterator = RALIIterator(pipe)
        cnt = 0
        for i, (images_array) in enumerate(imageIterator):
            print("INSIDE ITERATOR")
            # print(images_array.size())
            # images_array = np.transpose(images_array, [0, 3, 1, 2])
            # images_array = np.transpose(images_array, [ 2, 0, 1])
            
            # # print("\n",i)
            # # print("lables_array",labels_array)
            # print("\n\nPrinted first batch with", (batch_size), "images!")
            for element in list(range(batch_size)):
                cnt = cnt + 1
                print("size of image_Array   ",images_array[element].__sizeof__())
                draw_patches(images_array[element],cnt,"cpu")
                break
        imageIterator.reset()

    print("###############################################    TF CLASSIFICATION    ###############################################")
    print("###############################################    SUCCESS              ###############################################")

    #     jpegs, labels = fn.readers.file(file_root=data_path)
    #     images = fn.decoders.image(jpegs,file_root=data_path, output_type=types.RGB, shard_id=0, num_shards=1, random_shuffle=False)
    #     brightend_images = fn.gamma_correction(images, rocal_tensor_layout=types.NHWC, rocal_tensor_output_type=types.UINT8)
    #     # brightend_images2 = fn.brightness(brightend_images)

    #     pipe.set_outputs(brightend_images)

    # pipe.build()
    # imageIterator = RALIClassificationIterator(pipe)
    # cnt = 0
    # for i , it in enumerate(imageIterator):
    #     print("************************************** i *************************************",i)
    #     for img in it[0]:
    #         print(img.shape)
    #         cnt = cnt + 1
    #         draw_patches(img, cnt, "cpu")

    print("*********************************************************************")

    
    exit(0)

    # Need to call pipe.run() instead of iterator now (pipe.run() name is changed to pipe.run_tensor())
    output_data_batch_1 = pipe.run()
    print("----------------------")
    print(len(output_data_batch_1))
    # labels = pipe.rocalGetImageLabels()
    # print(labels)
    # output_data_batch_1 = pipe.run()
    # print("----------------------")
    # print(len(output_data_batch_1))

    # print("\n OUTPUT DATA!!!!: ", output_data_batch_1) # rocALTensorList 1
    exit(0)
    print("\n rocALTensor:: ",output_data_batch_1[0])
    # print("\n rocALTensor:: ",output_data_batch_1[0].at(0))
    print("\n size of rocalTensor ",output_data_batch_1[0].at(0).shape)

    print(output_data_batch_1[0].at(0).transpose(2,1,0).shape)
    print(output_data_batch_1[0].at(0))

    cv2.imwrite("output_images0_0.jpg", cv2.cvtColor(output_data_batch_1[0].at(0), cv2.COLOR_RGB2BGR))
    cv2.imwrite("output_images0_1.jpg", cv2.cvtColor(output_data_batch_1[0].at(1), cv2.COLOR_RGB2BGR))
    cv2.imwrite("output_images0_2.jpg", cv2.cvtColor(output_data_batch_1[0].at(2), cv2.COLOR_RGB2BGR))

    cv2.imwrite("output_images1_0.jpg", cv2.cvtColor(output_data_batch_1[1].at(0), cv2.COLOR_RGB2BGR))
    cv2.imwrite("output_images1_1.jpg", cv2.cvtColor(output_data_batch_1[1].at(1), cv2.COLOR_RGB2BGR))
    cv2.imwrite("output_images1_2.jpg", cv2.cvtColor(output_data_batch_1[1].at(2), cv2.COLOR_RGB2BGR))

    cv2.imwrite("output_images2_0.jpg", cv2.cvtColor(output_data_batch_1[2].at(0), cv2.COLOR_RGB2BGR))
    cv2.imwrite("output_images2_1.jpg", cv2.cvtColor(output_data_batch_1[2].at(1), cv2.COLOR_RGB2BGR))
    cv2.imwrite("output_images2_2.jpg", cv2.cvtColor(output_data_batch_1[2].at(2), cv2.COLOR_RGB2BGR))

    output_data_batch_2 = pipe.run()
    print("\n OUTPUT DATA BATCH 2!!!!: ", output_data_batch_2) # rocALTensorList 2

    cv2.imwrite("output_images2_0_0.jpg", cv2.cvtColor(output_data_batch_2[0].at(0), cv2.COLOR_RGB2BGR))
    cv2.imwrite("output_images2_0_1.jpg", cv2.cvtColor(output_data_batch_2[0].at(1), cv2.COLOR_RGB2BGR))
    cv2.imwrite("output_images2_0_2.jpg", cv2.cvtColor(output_data_batch_2[0].at(2), cv2.COLOR_RGB2BGR))

    cv2.imwrite("output_images2_1_0.jpg", cv2.cvtColor(output_data_batch_2[1].at(0), cv2.COLOR_RGB2BGR))
    cv2.imwrite("output_images2_1_1.jpg", cv2.cvtColor(output_data_batch_2[1].at(1), cv2.COLOR_RGB2BGR))
    cv2.imwrite("output_images2_1_2.jpg", cv2.cvtColor(output_data_batch_2[1].at(2), cv2.COLOR_RGB2BGR))

    cv2.imwrite("output_images2_2_0.jpg", cv2.cvtColor(output_data_batch_2[2].at(0), cv2.COLOR_RGB2BGR))
    cv2.imwrite("output_images2_2_1.jpg", cv2.cvtColor(output_data_batch_2[2].at(1), cv2.COLOR_RGB2BGR))
    cv2.imwrite("output_images2_2_2.jpg", cv2.cvtColor(output_data_batch_2[2].at(2), cv2.COLOR_RGB2BGR))

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
