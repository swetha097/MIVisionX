from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import itertools

import torch
import numpy as np
from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import amd.rocal.types as types
# import rali_pybind.tensor
import sys
import cv2
import os
import math
from math import sqrt
import ctypes

class ROCALCOCOIterator(object):
    """
    COCO ROCAL iterator for pyTorch.
    Parameters
    ----------
    pipelines : list of amd.rocal.pipeline.Pipeline
                List of pipelines to use
    size : int
           Epoch size.
    """

    def __init__(self, pipelines, tensor_layout=types.NCHW, reverse_channels=False, multiplier=None, offset=None, tensor_dtype=types.FLOAT, device="cpu", display=False, num_anchors=120087):

        try:
            assert pipelines is not None, "Number of provided pipelines has to be at least 1"
        except Exception as ex:
            print(ex)
        self.loader = pipelines
        self.tensor_format = tensor_layout
        self.multiplier = multiplier if multiplier else [1.0, 1.0, 1.0]
        self.offset = offset if offset else [0.0, 0.0, 0.0]
        self.reverse_channels = reverse_channels
        self.tensor_dtype = tensor_dtype
        print("\nself.tensor_dtype : ", self.tensor_dtype)
        self.device = device
        print(self.device)
        self.device_id = self.loader._device_id
        self.bs = self.loader._batch_size
        self.num_anchors = num_anchors
        #Number of batch size handled by each GPU
        self.len = math.ceil(self.loader.getRemainingImages()/self.bs)
        # Image sizes of a batch
        self.img_size = np.zeros((self.bs * 3), dtype="int32")

    def next(self):
        return self.__next__()

    def __next__(self):
        if(self.loader.isEmpty()):
            raise StopIteration
        if self.loader.rocalRun() != 0:
            raise StopIteration
        else:
            self.output_tensor_list = self.loader.rocalGetOutputTensors()

        self.w = self.output_tensor_list[0].batch_width()
        self.h = self.output_tensor_list[0].batch_height()
        self.bs = self.output_tensor_list[0].batch_size()
        self.color_format = self.output_tensor_list[0].color_format()

        torch_gpu_device = torch.device('cuda', self.device_id)

        if self.tensor_format == types.NCHW:
            self.out = torch.empty((self.bs, self.color_format, self.h, self.w,), dtype=torch.float32, device=torch_gpu_device)
        else:
            self.out = torch.empty((self.bs, self.h, self.w, self.color_format,), dtype=torch.float32, device=torch_gpu_device)
        self.output_tensor_list[0].copy_data(ctypes.c_void_p(self.out.data_ptr()))

        labels_array = self.loader.rocalGetBoundingBoxLabel()
        encodded_labels_tensor = []
        encoded_bboxes_tensor = []
        for label in labels_array:
            self.encoded_labels = torch.as_tensor(label, dtype=torch.int64)
            encodded_labels_tensor.append(self.encoded_labels)

        boxes_array = self.loader.rocalGetBoundingBoxCords()
        for box in boxes_array:
            self.encoded_bboxes = torch.as_tensor(box, dtype=torch.float16)
            self.encoded_bboxes = self.encoded_bboxes * 800
            self.encoded_bboxes = self.encoded_bboxes.view(-1, 4)
            encoded_bboxes_tensor.append(self.encoded_bboxes)

        matched_idxs = self.loader.rocalGetMatchedIndices()
        self.matched_idxs = torch.as_tensor(matched_idxs, dtype=torch.int64)
        matched_idxs_tensor = self.matched_idxs.view(-1, 120087)

        # Image sizes and Image id of a batch
        self.loader.GetImgSizes(self.img_size)
        image_size_tensor = torch.tensor(self.img_size).view(self.bs, 3)

        targets = { 'boxes' : encoded_bboxes_tensor,
                    'labels' : encodded_labels_tensor,
                    'image_id' : image_size_tensor[:, 2:3].cuda(),
                    'original_image_size' : image_size_tensor[:, 0:2].cuda(),
                    'matched_idxs' : matched_idxs_tensor
        }
        return self.out, targets

    def reset(self):
        self.loader.rocalResetLoaders()

    def __iter__(self):
        return self

def draw_patches(img, idx, bboxes, device):
    import cv2
    if device == "cpu":
            image = img.detach().numpy()
    else:
        image = img.cpu().numpy()
    image = image.transpose([1, 2, 0])
    _, htot, wtot = img.shape

    for (l, t, r, b) in bboxes:
        loc_ = [l, t, r, b]
        color = (255, 0, 0)
        thickness = 2
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.UMat(image).get()
        image = cv2.rectangle(image, (int(loc_[0]), int(loc_[1])), (int(
            (loc_[2])), int((loc_[3]))), color, thickness)
        cv2.imwrite("OUTPUT_IMAGES_PYTHON/NEW_API/COCO_READER/" + str(idx)+"_"+"train"+".png", image * 255)

def main():
    if  len(sys.argv) < 3:
        print ('Please pass image_path annotation_path cpu/gpu batch_size')
        exit(0)
    try:
        path= "OUTPUT_IMAGES_PYTHON/NEW_API/COCO_READER/"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
    except OSError as error:
        print(error)
    image_path = sys.argv[1]
    annotation_path = sys.argv[2]
    device = sys.argv[3]
    if(device == "cpu"):
        _rali_cpu = True
    else:
        _rali_cpu = False
    batch_size = int(sys.argv[4])
    num_threads = 1
    device_id = 0
    random_seed = random.SystemRandom().randint(0, 2**32 - 1)
    crop=800

    local_rank = 0
    world_size = 1

    rali_device = 'gpu'
    decoder_device = 'mixed'
    device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
    host_memory_padding = 140544512 if decoder_device == 'mixed' else 0

    # Anchors - load default anchors from a text file     
    with open('Default_anchors_retinanet.txt', 'r') as f_read:
        anchors = f_read.readlines()
    anchor_list = [float(x.strip())/800 for x in anchors]
    f_read.close()

    print("*********************************************************************")

    coco_train_pipeline = Pipeline(batch_size = batch_size, num_threads = num_threads, device_id = device_id,seed = random_seed, rocal_cpu = _rali_cpu, tensor_layout = types.NCHW)

    with coco_train_pipeline:
        jpegs, bboxes, labels = fn.readers.coco(file_root=image_path,
                                                 annotations_file=annotation_path, 
                                                 random_shuffle=True,
                                                 shard_id=local_rank, 
                                                 num_shards=world_size,
                                                 seed=random_seed, 
                                                 is_box_encoder=False,
                                                 is_box_iou_matcher=True)

        print("*********************** SHARD ID ************************",local_rank)
        print("*********************** NUM SHARDS **********************",world_size)
        images_decoded = fn.decoders.image(jpegs, device=decoder_device, output_type = types.RGB, file_root=image_path, annotations_file=annotation_path, random_shuffle=False,shard_id=local_rank, num_shards=world_size)
        flip_coin = fn.random.coin_flip(probability=0.5)
        images = fn.resize_mirror_normalize(images_decoded, device="gpu",
                                            image_type=types.RGB,
                                            resize_width=crop, resize_height=crop,
                                            mirror=flip_coin,
                                            rocal_tensor_layout = types.NCHW,
                                            rocal_tensor_output_type = types.FLOAT,
                                            mean=[0,0,0],
                                            std=[1,1,1])
        matched_idxs  = fn.box_iou_matcher(anchors=anchor_list, criteria=0.5,
                                     high_threshold=0.5, low_threshold=0.4,
                                     allow_low_quality_matches=True)

        coco_train_pipeline.set_outputs(images)
    coco_train_pipeline.build()
    COCOIteratorPipeline = ROCALCOCOIterator(coco_train_pipeline)
    cnt = 0
    for epoch in range(1):
        print("+++++++++++++++++++++++++++++EPOCH+++++++++++++++++++++++++++++++++++++",epoch)
        for i, it in enumerate(COCOIteratorPipeline):
            print("************************************** i *************************************",i)
            print(it)
            for i, img in enumerate(it[0]):
                draw_patches(img, cnt, it[1]['boxes'][i], "gpu")
                cnt =  cnt + 1
        COCOIteratorPipeline.reset()
    print("*********************************************************************")

if __name__ == '__main__':
    main()
