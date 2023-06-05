# Copyright (c) 2018 - 2023 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import torch
from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import amd.rocal.types as types
import sys
import os
import math
import ctypes
import numpy as np
import random

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
        self.device = device
        self.device_id = self.loader._device_id
        self.bs = self.loader._batch_size
        self.num_anchors = num_anchors
        # Image sizes of a batch
        self.img_size = np.zeros((self.bs * 2), dtype="int32")

    def next(self):
        return self.__next__()

    def __next__(self):
        if(self.loader.isEmpty()):
            raise StopIteration
        if self.loader.rocalRun() != 0:
            raise StopIteration
        self.lis = []  # Empty list for bboxes
        self.lis_lab = []  # Empty list of labels

        self.loader.copyToExternalTensor(
            self.out, self.multiplier, self.offset, self.reverse_channels, self.tensor_format, self.tensor_dtype)

        # Image id of a batch of images
        self.loader.GetImageId(self.image_id)
        # Image sizes of a batch
        self.loader.GetImgSizes(self.img_size)
        # Count of labels/ bboxes in a batch
        self.count_batch = self.loader.GetBoundingBoxCount(
            self.bboxes_label_count)
        # 1D labels & bboxes array
        if self.device == "cpu":
          self.encoded_bboxes = np.zeros((self.count_batch*4), dtype="float32")
          self.encoded_labels = np.zeros(self.count_batch, dtype="int32")
          self.loader.copyEncodedBoxesAndLables(self.encoded_bboxes, self.encoded_labels)
          encoded_bboxes_tensor = torch.tensor(self.encoded_bboxes).view(self.bs, -1, 4).contiguous()
          encodded_labels_tensor = torch.tensor(self.encoded_labels).long().view(self.bs, -1)
        else:
            self.output_tensor_list = self.loader.getOutputTensors()
        if self.out is None:
            self.w, self.h = self.output_tensor_list[0].max_shape()
            self.bs = self.output_tensor_list[0].batch_size()
            self.color_format = self.output_tensor_list[0].color_format() if self.color_format else self.color_format
            torch_gpu_device = torch.device('cuda', self.device_id)
            if self.tensor_format == types.NCHW:
                torch_gpu_device = torch.device('cuda', self.device_id)
                if self.tensor_dtype == types.FLOAT:
                    self.out = torch.empty((self.batch_size, self.color_format, self.h, self.w,), dtype = torch.float32, device = torch_gpu_device)
                elif self.tensor_dtype == types.FLOAT16:
                    self.out = torch.empty((self.batch_size, self.color_format, self.h, self.w,), dtype = torch.float16, device = torch_gpu_device)
                elif self.tensor_dtype == types.UINT8:
                    self.out = torch.empty((self.batch_size, self.color_format, self.h, self.w,), dtype = torch.uint8, device = torch_gpu_device)
            else:  # NHWC
                torch_gpu_device = torch.device('cuda', self.device_id)
                if self.tensor_dtype == types.FLOAT:
                    self.out = torch.empty((self.batch_size, self.h, self.w, self.color_format), dtype = torch.float32, device = torch_gpu_device)
                elif self.tensor_dtype == types.FLOAT16:
                    self.out = torch.empty((self.batch_size, self.h, self.w, self.color_format), dtype = torch.float16, device = torch_gpu_device)
                elif self.tensor_dtype == types.UINT8:
                    self.out = torch.empty((self.batch_size, self.h, self.w, self.color_format), dtype = torch.uint8, device = torch_gpu_device)
            self.labels_tensor = torch.empty(self.batch_size, dtype = torch.int32, device = torch_gpu_device)

        self.output_tensor_list[0].copy_data(ctypes.c_void_p(self.out.data_ptr()))

        labels_array = self.loader.GetBoundingBoxLabels()
        encodded_labels_tensor = []
        encoded_bboxes_tensor = []
        for label in labels_array:
            self.encoded_labels = torch.as_tensor(label, dtype=torch.int64)
            encodded_labels_tensor.append(self.encoded_labels)

        boxes_array = self.loader.GetBoundingBoxCords()
        for box in boxes_array:
            self.encoded_bboxes = torch.as_tensor(box, dtype=torch.float16)
            self.encoded_bboxes = self.encoded_bboxes * 800
            self.encoded_bboxes = self.encoded_bboxes.view(-1, 4)
            encoded_bboxes_tensor.append(self.encoded_bboxes)

        matched_idxs = self.loader.GetMatchedIndices()
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

def draw_patches(img, idx, bboxes, device="cpu"):
    import cv2
    if device == "cpu":
        image = img.detach().numpy()
    else:
        image = img.cpu().numpy()
    image = image.transpose([1, 2, 0])

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
    # Anchors

    def coco_anchors():  # Should be Tensor of floats in ltrb format - input - Mx4 where M="No of anchor boxes"
        fig_size = 300
        feat_size = [38, 19, 10, 5, 3, 1]
        steps = [8, 16, 32, 64, 100, 300]

        # use the scales here: https://github.com/amdegroot/ssd.pytorch/blob/master/data/config.py
        scales = [21, 45, 99, 153, 207, 261, 315]
        aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        default_boxes = []
        fk = fig_size/np.array(steps)
        # size of feature and number of feature
        for idx, sfeat in enumerate(feat_size):

            sk1 = scales[idx]/fig_size
            sk2 = scales[idx+1]/fig_size
            sk3 = sqrt(sk1*sk2)
            all_sizes = [(sk1, sk1), (sk3, sk3)]

            for alpha in aspect_ratios[idx]:
                w, h = sk1*sqrt(alpha), sk1/sqrt(alpha)
                all_sizes.append((w, h))
                all_sizes.append((h, w))
            for w, h in all_sizes:
                for i, j in itertools.product(range(sfeat), repeat=2):
                    cx, cy = (j+0.5)/fk[idx], (i+0.5)/fk[idx]
                    default_boxes.append((cx, cy, w, h))
        dboxes = torch.tensor(default_boxes, dtype=torch.float)
        dboxes.clamp_(min=0, max=1)
        # For IoU calculation
        dboxes_ltrb = dboxes.clone()
        dboxes_ltrb[:, 0] = dboxes[:, 0] - 0.5 * dboxes[:, 2]
        dboxes_ltrb[:, 1] = dboxes[:, 1] - 0.5 * dboxes[:, 3]
        dboxes_ltrb[:, 2] = dboxes[:, 0] + 0.5 * dboxes[:, 2]
        dboxes_ltrb[:, 3] = dboxes[:, 1] + 0.5 * dboxes[:, 3]

        return dboxes_ltrb
    default_boxes = coco_anchors().numpy().flatten().tolist()
    # Create Pipeline instance
    if args.display:
        pipe = Pipeline(batch_size=batch_size, num_threads=num_threads,
                        device_id=args.local_rank, seed=random_seed, rocal_cpu=rocal_cpu, tensor_layout=tensor_format, tensor_dtype=tensor_dtype)
    else:
        pipe = Pipeline(batch_size=batch_size, num_threads=num_threads,mean=[0.485*255, 0.456*255, 0.406*255],
                        std=[0.229*255, 0.224 * 255, 0.225*255], device_id=args.local_rank, seed=random_seed, rocal_cpu=rocal_cpu, tensor_layout=tensor_format, tensor_dtype=tensor_dtype)
    # Use pipeline instance to make calls to reader, decoder & augmentation's
    with pipe:
        jpegs, bboxes, labels = fn.readers.coco(
            file_root=image_path, annotations_file=annotation_path, random_shuffle=False, seed=random_seed)
        crop_begin, crop_size, bboxes, labels = fn.random_bbox_crop(bboxes, labels,
                                                                    device="cpu",
                                                                    aspect_ratio=[
                                                                        0.5, 2.0],
                                                                    thresholds=[
                                                                        0, 0.1, 0.3, 0.5, 0.7, 0.9],
                                                                    scaling=[
                                                                        0.3, 1.0],
                                                                    bbox_layout="xyXY",
                                                                    allow_no_crop=True,
                                                                    num_attempts=50)
        images_decoded = fn.decoders.image_slice(jpegs, crop_begin, crop_size, device="mixed", output_type=types.RGB, file_root=image_path,
                                                 annotations_file=annotation_path, random_shuffle=False, seed=random_seed, num_shards=world_size, shard_id=local_rank)
        res_images = fn.resize(images_decoded, resize_x=300, resize_y=300)
        saturation = fn.uniform(rng_range=[0.5, 1.5])
        contrast = fn.uniform(rng_range=[0.5, 1.5])
        brightness = fn.uniform(rng_range=[0.875, 1.125])
        hue = fn.uniform(rng_range=[-0.5, 0.5])
        ct_images = fn.color_twist(
            res_images, saturation=saturation, contrast=contrast, brightness=brightness, hue=hue)
        flip_coin = fn.random.coin_flip(probability=0.5)
        bboxes = fn.bb_flip(bboxes, ltrb=True, horizontal=flip_coin)
        cmn_images = fn.crop_mirror_normalize(ct_images,
                                                  crop=(300, 300),
                                                  mean=[0, 0, 0],
                                                  std=[1, 1, 1],
                                                  mirror=flip_coin,
                                                  output_dtype=types.UINT8,
                                                  output_layout=types.NHWC,
                                                  pad_output=False)
        if args.display:
            _, _ = fn.box_encoder(bboxes, labels,
                                  criteria=0.5,
                                  anchors=default_boxes)
        else:
            _, _ = fn.box_encoder(bboxes, labels,
                                  criteria=0.5,
                                  anchors=default_boxes,
                                  offset=True, stds=[0.1, 0.1, 0.2, 0.2], scale=300)

        pipe.set_outputs(cmn_images)
    # Build the pipeline
    pipe.build()
    # Dataloader
    if(args.rocal_gpu):
        data_loader = ROCALCOCOIterator(
            pipe, multiplier=pipe._multiplier, offset=pipe._offset, display=display, tensor_layout=tensor_format, tensor_dtype=tensor_dtype, device="cuda", num_anchors=len(default_boxes)/4)

    else:
        data_loader = ROCALCOCOIterator(
            pipe, multiplier=pipe._multiplier, offset=pipe._offset, display=display, tensor_layout=tensor_format, tensor_dtype=tensor_dtype, device="cpu", num_anchors=len(default_boxes)/4)

    import timeit
    start = timeit.default_timer()
    # Enumerate over the Dataloader
    for epoch in range(int(args.num_epochs)):
        print("EPOCH:::::", epoch)
        for i, it in enumerate(data_loader, 0):
            if i == 0:
                print("**************", i, "*******************")
                print("**************starts*******************")
                print("\n IMAGES : \n", it[0])
                print("\nBBOXES:\n", it[1])
                print("\nLABELS:\n", it[2])
                print("\nIMAGE ID:\n", it[3])
                print("\nIMAGE SIZE:\n", it[4])
                print("**************ends*******************")
                print("**************", i, "*******************")
        data_loader.reset()
    #Your statements here
    stop = timeit.default_timer()

    print('\n Time: ', stop - start)

    print("###############################################    COCO READER    ###############################################")
    print("###############################################    SUCCESS        ###############################################")


if __name__ == '__main__':
    main()
