from nvidia.dali.pipeline import Pipeline
import nvidia.dali.types as types
import nvidia.dali.fn as fn
import nvidia.dali.math as math
import numpy as np
from time import time
import os.path
import cv2
import random
import glob
random.seed(1231231)   # Random is used to pick colors

#file_root = "/home/mcw/fiona/coco_10_img/train_2017"
#annotations_file = "/home/mcw/fiona/coco_10_img/annotations/instances_train2017.json"
file_root = "/media/kamal/coco_meta_reader/MIVisionX/utilities/rocAL/rocAL_unittests/MIVisionX-data/mini_coco_dataset/coco_test_4/val_2017_small"
annotations_file = "/media/kamal/coco_meta_reader/MIVisionX/utilities/rocAL/rocAL_unittests/MIVisionX-data/mini_coco_dataset/coco_test_4/annotations/instances_val2017_small.json"
def coco_reader_def():
    inputs, bboxes, labels, pixelwise_masks, image_ids = fn.readers.coco(
        file_root=file_root,
        annotations_file=annotations_file,
        pixelwise_masks=True, # Load segmentation mask data as polygons
        ratio=False,         # Bounding box and mask polygons to be expressed in relative coordinates
        ltrb=False,          # Bounding boxes to be expressed as left, top, right, bottom coordinates
        image_ids = True,
    )
    images = fn.decoders.image(inputs, device='cpu')
    return images, bboxes, labels, pixelwise_masks,image_ids

pipe = Pipeline(batch_size=4, num_threads=1, device_id=None)
with pipe:
    inputs, bboxes, labels, pixelwise_masks,image_ids = coco_reader_def()
    pipe.set_outputs(inputs, bboxes, labels, pixelwise_masks, image_ids)
pipe.build()
outputs = pipe.run()

file_list = sorted(glob.glob(file_root+"/*.jpg"))
print (file_list)
for i in range(0, len(file_list)):
    print (file_list[i])
    #img = cv2.imread(file_list[i])
    #print (img.shape)
    inputs = outputs[0].at(i)
    #print (inputs.shape)
    labels = outputs[2].at(i)
    #print (labels)
    polygons = outputs[3].at(i)
    print (polygons.shape)
    h,w,c = polygons.shape
    #cv2.imwrite("polygons/"+str(i)+".jpg",polygons)
    image_ids = outputs[4].at(i)
    #print (image_ids)
    fp = open(os.path.basename(file_list[i])+".txt", 'w')
    for j in range(0,h):
        for k in range(0,w):
            fp.write(str(polygons[j][k][0])+"\n")
    fp.close()

