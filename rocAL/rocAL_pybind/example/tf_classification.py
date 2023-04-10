from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import sys
import cv2
import os
import tensorflow as tf
from amd.rocal.plugin.tf import ROCALIterator
from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import amd.rocal.types as types
import tensorflow as tf
import numpy as np
from PIL import Image as im




def draw_patches(img, idx, device):
    print("DRAW PATCHES!!")
    #image is expected as a tensor, bboxes as numpy
    import cv2 
    print("IN DRAW_PATCH  ",img.shape)
    image =img
    print(image.shape,type(image))
    
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("OUTPUT_IMAGES_PYTHON/NEW_API/TF_READER/CLASSIFICATION/" + str(idx)+"_"+"train"+".png", image1 )
def main():
    if  len(sys.argv) < 1:
        print ('Please pass image_folder cpu/gpu batch_size')
        exit(0)
    try:
        path= "OUTPUT_IMAGES_PYTHON/NEW_API/TF_READER/CLASSIFICATION/"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
    except OSError as error:   
        print(error)
    data_path = sys.argv[1]
    if(sys.argv[2] == "cpu"):
        rocalCPU = True
    else:
        rocalCPU = False
    batch_size = int(sys.argv[3])
    print("batch_size  ",batch_size)
    TFRecordReaderType = 0
    featureKeyMap = {
        'image/encoded':'image/encoded',
        'image/class/label':'image/class/label',
        'image/filename':'image/filename'
    }
    
    num_threads = 1
    device_id = 0
    random_seed = random.SystemRandom().randint(0, 2**32 - 1)
    crop=300
    oneHotLabel = 1
    local_rank = 0
    world_size = 1
    rali_cpu= False
    rali_device = 'cpu' if rali_cpu else 'gpu'
    decoder_device = 'cpu' if rali_cpu else 'mixed'
    device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
    host_memory_padding = 140544512 if decoder_device == 'mixed' else 0

    # Create Pipeline instance
    pipe = Pipeline(batch_size=batch_size, num_threads=num_threads,device_id=device_id, seed=2, rocal_cpu=rocalCPU)
    # Use pipeline instance to make calls to reader, decoder & augmentation's
    with pipe:
        inputs = fn.readers.tfrecord(path=data_path, index_path = "", reader_type=TFRecordReaderType, user_feature_key_map=featureKeyMap,
            features={
                'image/encoded':tf.io.FixedLenFeature((), tf.string, ""),
                'image/class/label':tf.io.FixedLenFeature([1], tf.int64,  -1),
                'image/filename':tf.io.FixedLenFeature((), tf.string, "")
            }
        )
        jpegs = inputs["image/encoded"]
        images = fn.decoders.image(jpegs, user_feature_key_map=featureKeyMap, output_type=types.RGB, path=data_path)
        resized = fn.resize(images, resize_width=300, resize_height=300, rocal_tensor_layout = types.NHWC, rocal_tensor_output_type = types.UINT8)
        cmnp = fn.crop_mirror_normalize(resized, device="cpu",
                                            rocal_tensor_layout = types.NHWC,
                                            rocal_tensor_output_type = types.FLOAT,
                                            output_dtype = types.FLOAT,
                                            crop=[300, 300],
                                            mirror=0,
                                            image_type=types.RGB,
                                            mean=[0,0,0],
                                            std=[1,1,1])
        # if(oneHotLabel == 1):
        #     print("check ")
        #     labels = inputs["image/class/label"]
        #     _ = fn.one_hot(labels, num_classes=1000)
        #     print("labels ",_)
        # bright = fn .brightness(resized)
        pipe.set_outputs(cmnp)

        # Build the pipeline
        pipe.build()
        # Dataloader
        imageIterator = ROCALIterator(pipe)
        cnt = 0
        for i, (images_array,labels_array) in enumerate(imageIterator):
            if 1:
                print("\n",i)
                print("lables_array",labels_array)
                print("\n\nPrinted first batch with", (batch_size), "images!")
            for element in list(range(batch_size)):
                print("element in unittest  ",element)
                cnt = cnt + 1
                print("size of image_Array   ",images_array[element].__sizeof__())
                draw_patches(images_array[element],cnt,"cpu")
                # break
        imageIterator.reset()

    print("###############################################    TF CLASSIFICATION    ###############################################")
    print("###############################################    SUCCESS              ###############################################")

if __name__ == '__main__':
    main()