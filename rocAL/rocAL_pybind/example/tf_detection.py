from __future__ import absolute_import

from __future__ import division
from __future__ import print_function
import random
import sys
import cv2
import os
import tensorflow as tf
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
    import cv2
    print("*************************************draw_patches**********************************")
    image =img
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("OUTPUT_IMAGES_PYTHON/NEW_API/TF_READER/DETECTION/"+str(idx)+"_"+"aki"+".png", image)
    htot, wtot ,_ = image.shape
    for (l, t, r, b) in bboxes:
        loc_ = [l, t, r, b]
        color = (255, 0, 0)
        thickness = 2
        # print("values",loc_[0]*wtot,loc_[1] * htot,loc_[2] * wtot,loc_[3] * htot,color,thickness)
        image = cv2.rectangle(image, (int(loc_[0]*wtot), int(loc_[1] * htot)), (int(
            (loc_[2] * wtot)), int((loc_[3] * htot))), color, thickness)
        cv2.imwrite("OUTPUT_IMAGES_PYTHON/NEW_API/TF_READER/DETECTION/"+str(idx)+"_"+"aki"+".png", image)
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
    
    pipe = Pipeline(batch_size=batch_size, num_threads=num_threads,device_id=device_id, seed=2, rocal_cpu=_rocal_cpu, tensor_dtype=types.FLOAT)
    print(pipe)
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
        images = fn.decoders.image_random_crop(jpegs,user_feature_key_map=featureKeyMap, output_type=types.RGB,
                                                      random_aspect_ratio=[0.8, 1.25],
                                                      random_area=[0.1, 1.0],
                                                      num_attempts=100,path = data_path)
                                                      
        
        resized = fn.resize(images, resize_width=400, resize_height=400,rocal_tensor_output_type = types.UINT8, rocal_tensor_layout = types.NHWC)
        cmnp = fn.crop_mirror_normalize(resized, device="cpu",
                                            rocal_tensor_layout = types.NHWC,
                                            rocal_tensor_output_type = types.FLOAT,
                                            crop=[300, 300],
                                            mirror=0,
                                            image_type=types.RGB,
                                            mean=[0,0,0],
                                            std=[1,1,1])
        pipe.set_outputs(cmnp)

        # Build the pipeline
        pipe.build()
        # Dataloader
        imageIterator = ROCALIterator(pipe)
        cnt = 0
    for i, (images_array, bboxes_array, labels_array, num_bboxes_array) in enumerate(imageIterator, 0):
        print("ROCAL augmentation pipeline - Processing batch %d....." % i)
        for element in list(range(batch_size)):
            cnt = cnt + 1
            # print("Processing image %d....." % element)
            features_dict = {
                "image": images_array[element],
                "true_image_shape": np.array([len(images_array[element]), len(images_array[element, 0]), len(images_array[element, 0, 0])])
            }
            labels_dict = {
                "num_groundtruth_boxes": num_bboxes_array[element],
                "groundtruth_boxes": bboxes_array[element],
                "groundtruth_classes": get_onehot(labels_array[element], numClasses),
                "groundtruth_weights": get_weights(num_bboxes_array[element])
            }
            processed_tensors = (features_dict, labels_dict)
            print("\nPROCESSED_TENSORS:\n", processed_tensors)
            draw_patches(images_array[element],cnt,bboxes_array[element])
        print("\n\nPrinted first batch with", (batch_size), "images!")
        break
    imageIterator.reset()

    print("###############################################    TF DETECTION    ###############################################")
    print("###############################################    SUCCESS              ###############################################")

if __name__ == '__main__':
    main()