from __future__ import absolute_import

from __future__ import division
from __future__ import print_function
import random

# from amd.rali.pipeline import Pipeline
# import amd.rali.fn as fn
# import amd.rali.types as types
# import rali_pybind.tensor
import sys
import cv2
import os
import tensorflow as tf
# from amd.rali.plugin.tf import ROCALIterator, ROCAL_iterator
# from amd.rali.pipeline import Pipeline
# import amd.rali.types as types
# import amd.rali.fn as fn

from amd.rocal.plugin.pytorch import ROCALClassificationIterator
from amd.rocal.plugin.tf import ROCALIterator

from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import amd.rocal.types as types

import tensorflow as tf
import numpy as np
from PIL import Image as im


def get_onehot(image_labels_array, numClasses):
    one_hot_vector_list = []
    for label in image_labels_array:
        one_hot_vector = np.zeros(numClasses)
        if label[0] != 0:
            np.put(one_hot_vector, label[0] - 1, 1)
        one_hot_vector_list.append(one_hot_vector)

    one_hot_vector_array = np.array(one_hot_vector_list)

    return one_hot_vector_array

def get_weights(num_bboxes):
    weights_array = np.zeros(100)
    for pos in list(range(num_bboxes)):
        np.put(weights_array, pos, 1)

    return weights_array


def draw_patches(img, idx, bboxes):
    #image is expected as a tensor, bboxes as numpy
    import cv2
    # image = img.detach().numpy()
    print("*************************************draw_patches**********************************")
    # image = img.transpose([0, 1, 2])
    image =img
    print("image shape in draw patch ",image.shape)
    print(image.dtype)
    # print("image",image)
    # image =image.astype(int)
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("OUTPUT_IMAGES_PYTHON/NEW_API/TF_READER/DETECTION/"+str(idx)+"_"+"aki"+".png", image)

    # image = cv2.normalize(image, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    htot, wtot ,_ = image.shape
    # htot, wtot =300,300
    print("shape",htot,wtot)
    # print("bboxes",bboxes)

    for (l, t, r, b) in bboxes:
        loc_ = [l, t, r, b]
        # print("loc_",loc_)
        print("ch1")
        color = (255, 0, 0)
        thickness = 2
        print("ch2")
        
        # image = cv2.UMat(image).get()
        print("ch3")
        
        print("valuessssss",loc_[0]*wtot,loc_[1] * htot,loc_[2] * wtot,loc_[3] * htot,color,thickness)
        image = cv2.rectangle(image, (int(loc_[0]*wtot), int(loc_[1] * htot)), (int(
            (loc_[2] * wtot)), int((loc_[3] * htot))), color, thickness)
        cv2.imwrite("OUTPUT_IMAGES_PYTHON/NEW_API/TF_READER/DETECTION/"+str(idx)+"_"+"aki"+".png", image)
        print("end of draw_patch")
def main():
    if  len(sys.argv) < 1:
        print ('Please pass image_folder cpu/gpu batch_size')
        exit(0)
    try:
        path= "OUTPUT_IMAGES_PYTHON/NEW_API/TF_READER/DETECTION/"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
    except OSError as error:   
        print(error)
    data_path = sys.argv[1]
    if(sys.argv[2] == "cpu"):
        _rocal_cpu = True
    else:
        _rocal_cpu = False
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
    numClasses = 91     
    
    random_seed = random.SystemRandom().randint(0, 2**32 - 1)
    crop=300

    local_rank = 0
    world_size = 1
    rocal_cpu= True
    rocal_device = 'cpu' if rocal_cpu else 'gpu'
    decoder_device = 'cpu' if rocal_cpu else 'mixed'
    device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
    host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
    
    pipe = Pipeline(batch_size=batch_size, num_threads=num_threads,device_id=device_id, seed=2, rocal_cpu=_rocal_cpu)

    with pipe:
        inputs = fn.readers.tfrecord(path=data_path, index_path = "", reader_type=TFRecordReaderType, user_feature_key_map=featureKeyMap,random_shuffle=True,
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
        # images = fn.decoders.image(jpegs, user_feature_key_map=featureKeyMap, output_type=types.RGB, path=data_path,random_shuffle=True)
        images = fn.decoders.image_random_crop(jpegs,user_feature_key_map=featureKeyMap, output_type=types.RGB,
                                                      random_aspect_ratio=[0.8, 1.25],
                                                      random_area=[0.1, 1.0],
                                                      num_attempts=100,path = data_path)
        resized = fn.resize(images, resize_width=600, resize_height=600,rocal_tensor_output_type = types.UINT8, rocal_tensor_layout = types.NHWC)
        # crop_image= fn.crop_mirror_normalize(images,crop_d=100,crop_h =100,device=1,rocal_tensor_output_type = types.UINT8, rocal_tensor_layout = types.NHWC)
        # bright = fn .brightness(images)

        pipe.set_outputs(resized)
        # pipe.set_outputs(images)

        # Build the pipeline
        pipe.build()
        # Dataloader
        imageIterator = ROCALIterator(pipe)
        cnt = 0
        print("imageIterator   ",imageIterator)
        for i, (images_array, bboxes_array, labels_array,num_bboxes_array) in enumerate(imageIterator, 0):
            print("images_array",images_array)
            # images_array = np.transpose(images_array, [0, 2, 3, 1])
            # print("images_array1",images_array)
        print("ROCAL augmentation pipeline - Processing batch %d....." % i)

        for element in list(range(batch_size)):
            cnt = cnt + 1
            print("image shape ",images_array[element].shape)
            # if args.print_tensor:
            print("Processing image %d....." % element)
            features_dict = {
                "image": images_array[element],
                "true_image_shape": np.array([len(images_array[element]), len(images_array[element, 0]), len(images_array[element, 0, 0])])
            }
            
            draw_patches(images_array[element],cnt,bboxes_array[element])

            labels_dict = {
                "num_groundtruth_boxes": num_bboxes_array[element],
                "groundtruth_boxes": bboxes_array[element],
                "groundtruth_classes": get_onehot(labels_array[element], numClasses),
                "groundtruth_weights": get_weights(num_bboxes_array[element])
            }
            processed_tensors = (features_dict, labels_dict)
            # if args.print_tensor:
            #     print("\nPROCESSED_TENSORS:\n", processed_tensors)
            draw_patches(images_array[element],cnt,bboxes_array[element])
        print("\n\nPrinted first batch with", (batch_size), "images!")
        # break
    imageIterator.reset()

    print("###############################################    TF DETECTION    ###############################################")
    print("###############################################    SUCCESS              ###############################################")

    #     jpegs, labels = fn.readers.file(file_root=data_path)
    #     images = fn.decoders.image(jpegs,file_root=data_path, output_type=types.RGB, shard_id=0, num_shards=1, random_shuffle=False)
    #     brightend_images = fn.gamma_correction(images, rocal_tensor_layout=types.NHWC, rocal_tensor_output_type=types.UINT8)
    #     # brightend_images2 = fn.brightness(brightend_images)

    #     pipe.set_outputs(brightend_images)

    # pipe.build()
    # imageIterator = ROCALClassificationIterator(pipe)
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