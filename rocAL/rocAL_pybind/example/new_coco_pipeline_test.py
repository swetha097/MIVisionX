from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from math import sqrt
import torch
import random
import itertools

from amd.rali.pipeline import Pipeline
import amd.rali.fn as fn
import amd.rali.types as types
import sys
import numpy as np

#Testing the fn import
# print(dir(fn))
# print(fn.__dict__)




class RALICOCOIterator(object):
    """
    COCO RALI iterator for pyTorch.

    Parameters
    ----------
    pipelines : list of amd.rali.pipeline.Pipeline
                List of pipelines to use
    size : int
           Epoch size.
    """

    def __init__(self, pipelines, tensor_layout=types.NCHW, reverse_channels=False, multiplier=None, offset=None, tensor_dtype=types.FLOAT, display=False):

        # self._num_gpus = len(pipelines)
        assert pipelines is not None, "Number of provided pipelines has to be at least 1"

        self.loader = pipelines
        self.tensor_format = tensor_layout
        self.multiplier = multiplier if multiplier else [1.0, 1.0, 1.0]
        self.offset = offset if offset else [0.0, 0.0, 0.0]
        self.reverse_channels = reverse_channels
        self.tensor_dtype = tensor_dtype
        self.bs = self.loader._batch_size
        self.w = self.loader.getOutputWidth()
        self.h = self.loader.getOutputHeight()
        self.n = self.loader.getOutputImageCount()
        self.rim = self.loader.getRemainingImages()
        self.display = display
        print("____________REMAINING IMAGES____________:", self.rim)
        color_format = self.loader.getOutputColorFormat()
        self.p = (1 if color_format is types.GRAY else 3)
        if self.tensor_dtype == types.FLOAT:
            self.out = np.zeros(
                (self.bs*self.n, self.p, int(self.h/self.bs), self.w,), dtype="float32")
        elif self.tensor_dtype == types.FLOAT16:
            self.out = np.zeros(
                (self.bs*self.n, self.p, int(self.h/self.bs), self.w,), dtype="float16")

    def next(self):
        return self.__next__()

    def __next__(self):
        print("In the next routine of COCO Iterator")
        if(self.loader.isEmpty()):
            timing_info = self.loader.Timing_Info()
            print("Load     time ::", timing_info.load_time)
            print("Decode   time ::", timing_info.decode_time)
            print("Process  time ::", timing_info.process_time)
            print("Transfer time ::", timing_info.transfer_time)
            raise StopIteration

        if self.loader.run() != 0:
            raise StopIteration
        self.lis = []  # Empty list for bboxes
        self.lis_lab = []  # Empty list of labels

        if(types.NCHW == self.tensor_format):
            self.loader.copyToTensorNCHW(
                self.out, self.multiplier, self.offset, self.reverse_channels, int(self.tensor_dtype))
        else:
            self.loader.copyToTensorNHWC(
                self.out, self.multiplier, self.offset, self.reverse_channels, int(self.tensor_dtype))

        self.img_names_length = np.empty(self.bs, dtype="int32")
        self.img_names_size = self.loader.GetImageNameLen(
            self.img_names_length)
        print("Image name length:", self.img_names_size)
# Images names of a batch
        self.Img_name = self.loader.GetImageName(self.img_names_size)
        print("Image names in a batch ", self.Img_name)
# Count of labels/ bboxes in a batch
        self.bboxes_label_count = np.zeros(self.bs, dtype="int32")
        self.count_batch = self.loader.GetBoundingBoxCount(
            self.bboxes_label_count)
        print("Count Batch:", self.count_batch)
# 1D labels array in a batch
        self.labels = np.zeros(self.count_batch, dtype="int32")
        self.loader.GetBBLabels(self.labels)
        print(self.labels)
# 1D bboxes array in a batch
        self.bboxes = np.zeros((self.count_batch*4), dtype="float32")
        self.loader.GetBBCords(self.bboxes)
        print(self.bboxes)
# Image sizes of a batch
        self.img_size = np.zeros((self.bs * 2), dtype="int32")
        self.loader.GetImgSizes(self.img_size)
        print("Image sizes:", self.img_size)
        count = 0
        sum_count = 0
        for i in range(self.bs):
            count = self.bboxes_label_count[i]
            print("labels:", self.labels[sum_count: sum_count+count])
            print("bboxes:", self.bboxes[sum_count*4: (sum_count+count)*4])
            print("Image w & h:", self.img_size[i*2:(i*2)+2])
            print("Image names:", self.Img_name[i*16:(i*16)+12])
            self.img_name = self.Img_name[i*16:(i*16)+12]
            self.img_name = self.img_name.decode('utf_8')
            self.img_name = np.char.lstrip(self.img_name, chars='0')
            print("Image names:", self.img_name)
            self.label_2d_numpy = (self.labels[sum_count: sum_count+count])
            if(self.loader._BoxEncoder != True):
                self.label_2d_numpy = np.reshape(
                    self.label_2d_numpy, (-1, 1)).tolist()
            self.bb_2d_numpy = (self.bboxes[sum_count*4: (sum_count+count)*4])
            self.bb_2d_numpy = np.reshape(self.bb_2d_numpy, (-1, 4)).tolist()
            # Draw images: make sure to revert the mean and std to 0 and 1 for displaying original images without normalization
            if self.display:
               img = torch.from_numpy(self.out)
               draw_patches(img[i], self.img_name, self.bb_2d_numpy)
            if(self.loader._BoxEncoder == True):
                
                # Converting from "xywh" to "ltrb" format ,
                # where the values of l, t, r, b always lie between 0 & 1
                # Box Encoder input & output:
                # input : N x 4 , "xywh" format
                # output : 8732 x 4 , "xywh" format and normalized
                htot, wtot = 1, 1
                bbox_sizes = []
                i=0
                for (l,t,r,b) in self.bb_2d_numpy:
                    

                    bbox_size = (l/wtot, t/htot, r/wtot, b/htot)
                    bbox_sizes.append(bbox_size)
                    i=i+1

                encoded_bboxes, encodded_labels = self.loader.encode(bboxes_in=bbox_sizes, labels_in=self.label_2d_numpy)
                if(self.loader._castLabels == True):
                    encodded_labels = encodded_labels.type(torch.FloatTensor)
                self.lis.append(encoded_bboxes)
                self.lis_lab.append(encodded_labels)
            else:
                self.lis_lab.append(self.label_2d_numpy)
                self.lis.append(self.bb_2d_numpy)
            sum_count = sum_count + count

        if (self.loader._BoxEncoder != True):
            self.target = self.lis
            self.target1 = self.lis_lab

            max_cols = max([len(row) for batch in self.target for row in batch])
            max_rows = max([len(batch) for batch in self.target])
            self.bb_padded = [
                batch + [[0] * (max_cols)] * (max_rows - len(batch)) for batch in self.target]
            self.bb_padded = torch.FloatTensor(
                [row + [0] * (max_cols - len(row)) for batch in self.bb_padded for row in batch])
            self.bb_padded = self.bb_padded.view(-1, max_rows, max_cols)
            # print(self.bb_padded)

            max_cols1 = max([len(row) for batch in self.target1 for row in batch])
            max_rows1 = max([len(batch) for batch in self.target1])
            self.labels_padded = [
                batch + [[0] * (max_cols1)] * (max_rows1 - len(batch)) for batch in self.target1]
            self.labels_padded = torch.LongTensor(
                [row + [0] * (max_cols1 - len(row)) for batch in self.labels_padded for row in batch])
            self.labels_padded = self.labels_padded.view(-1, max_rows1, max_cols1)
            # print(self.labels_padded)
        else:
            self.bb_padded = torch.stack(self.lis)
            self.labels_padded = torch.stack(self.lis_lab)

        if self.tensor_dtype == types.FLOAT:
            return torch.from_numpy(self.out), self.bb_padded, self.labels_padded
        elif self.tensor_dtype == types.FLOAT16:
            return torch.from_numpy(self.out.astype(np.float16)), self.bb_padded, self.labels_padded

    def reset(self):
        self.loader.raliResetLoaders()

    def __iter__(self):
        return self

def draw_patches(img,idx, bboxes):
    #image is expected as a tensor, bboxes as numpy
    import cv2
    image = img.detach().numpy()
    image = image.transpose([1,2,0])
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR )
 
    _,htot ,wtot = img.shape
    image = cv2.UMat(image).get()
    cv2.imwrite(str(idx)+"_"+"train"+".png", image)

def main():
    if len(sys.argv) < 5:
        print('Please pass the folder image_folder Annotation_file cpu/gpu batch_size display(True/False)')
        exit(0)

    image_path = sys.argv[1]
    ann_path = sys.argv[2]
    if(sys.argv[3] == "cpu"):
        _rali_cpu = True
    else:
        _rali_cpu = False
    bs = int(sys.argv[4])
    display = sys.argv[5]
    nt = 1
    di = 0
    crop_size = 300
    random_seed = random.SystemRandom().randint(0, 2**32 - 1)

    pipe = Pipeline(batch_size=bs, num_threads=1,device_id=0, seed=2, rali_cpu=_rali_cpu)

    with pipe:
        jpegs, bb, labels = fn.readers.coco(
            file_root=image_path, annotations_file=ann_path, random_shuffle=True, seed=1)
        images_decoded = fn.decoders.image(jpegs, output_type=types.RGB)
        res_images = fn.resize(images_decoded, resize_x=300, resize_y=300)
        flip_coin = fn.random.coin_flip(probability=0.5)
        if not display:
            images = fn.crop_mirror_normalize(res_images,
                                        crop=(300, 300),
                                        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                                        mirror=flip_coin,
                                        output_dtype=types.FLOAT,
                                        output_layout=types.NCHW,
                                        pad_output=False)
        else:
            images = fn.crop_mirror_normalize(res_images,
                                        crop=(300, 300),
                                        mean=[0,0,0],
                                        std=[1,1,1],
                                        mirror=flip_coin,
                                        output_dtype=types.FLOAT,
                                        output_layout=types.NCHW,
                                        pad_output=False)
        saturation = fn.uniform(range=[0.5, 1.5])
        contrast = fn.uniform(range=[0.5, 1.5])
        brightness = fn.uniform(range=[0.875, 1.125])
        hue = fn.uniform(range=[-0.5, 0.5])
        images = fn.color_twist(images, saturation=saturation, contrast=contrast, brightness=brightness, hue=hue)
        pipe.set_outputs(images, bb , labels)

    data_loader = RALICOCOIterator(
        pipe, multiplier=pipe._multiplier, offset=pipe._offset,display=display)
    epochs = 1
    for epoch in range(int(epochs)):
        print("EPOCH:::::",epoch)
        for i, it in enumerate(data_loader, 0):
            print("**************", i, "*******************")
            print("**************starts*******************")
            print("\n IMAGES : \n", it[0])
            print("\nBBOXES:\n", it[1])
            print("\nLABELS:\n", it[2])
            print("**************ends*******************")
            print("**************", i, "*******************")
        data_loader.reset()


if __name__ == '__main__':
    main()


    



