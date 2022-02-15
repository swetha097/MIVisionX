from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from math import sqrt
import torch
import cv2
import random

from amd.rali.pipeline import Pipeline
import amd.rali.fn as fn
import amd.rali.types as types
import sys
import numpy as np

def get_val_dataset():
    labels_dict =  {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 13, 13: 14, 14: 15, 15: 16, 16: 17, 17: 18, 18: 19, 19: 20, 20: 21, 21: 22, 22: 23, 23: 24, 24: 25, 25: 27, 26: 28, 27: 31, 28: 32, 29: 33, 30: 34, 31: 35, 32: 36, 33: 37, 34: 38, 35: 39, 36: 40, 37: 41, 38: 42, 39: 43, 40: 44, 41: 46, 42: 47, 43: 48, 44: 49, 45: 50, 46: 51, 47: 52, 48: 53, 49: 54, 50: 55, 51: 56, 52: 57, 53: 58, 54: 59, 55: 60, 56: 61, 57: 62, 58: 63, 59: 64, 60: 65, 61: 67, 62: 70, 63: 72, 64: 73, 65: 74, 66: 75, 67: 76, 68: 77, 69: 78, 70: 79, 71: 80, 72: 81, 73: 82, 74: 84, 75: 85, 76: 86, 77: 87, 78: 88, 79: 89, 80: 90}
    return labels_dict

class RALIGenericTrainIterator(object):
    """
    COCO RALI iterator for pyTorch.
    Parameters
    ----------
    pipelines : list of amd.rali.pipeline.Pipeline
                List of pipelines to use
    size : int
           Epoch size.
    """

    def __init__(self, pipelines, tensor_layout=types.NCHW, reverse_channels=False, multiplier=[1.0, 1.0, 1.0], offset=[0.0, 0.0, 0.0], tensor_dtype=types.FLOAT16):

        assert pipelines is not None, "Number of provided pipelines has to be at least 1"

        self.loader = pipelines
        self.iter = 0
        self.tensor_format = tensor_layout
        self.multiplier = multiplier
        self.offset = offset
        self.reverse_channels = reverse_channels
        self.tensor_dtype = tensor_dtype
        self.bs = self.loader._batch_size
        self.w = self.loader.getOutputWidth()
        self.h = self.loader.getOutputHeight()
        self.n = self.loader.getOutputImageCount()
        self.rim = self.loader.getRemainingImages()
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
        self.img_names_size = self.loader.GetImageNameLen(self.img_names_length)
# Images names of a batch
        self.Img_name = self.loader.GetImageName(self.img_names_size)
#Count of labels/ bboxes in a batch
        self.bboxes_label_count = np.zeros(self.bs, dtype="int32")
        self.count_batch = self.loader.GetBoundingBoxCount(self.bboxes_label_count)
        #print("self.bboxes_label_count", self.bboxes_label_count)
# 1D labels array in a batch
        self.labels = np.zeros(self.count_batch, dtype="int32")
        self.loader.GetBBLabels(self.labels)
# 1D bboxes array in a batch
        self.bboxes = np.zeros((self.count_batch*4), dtype="float32")
        self.loader.GetBBCords(self.bboxes)
#Image sizes of a batch
        self.img_size = np.zeros((self.bs * 2),dtype = "int32")
        self.loader.GetImgSizes(self.img_size)
#Image ROI width and height
        self.roi_width = np.zeros((self.bs),dtype = "uint32")
        self.roi_height = np.zeros((self.bs),dtype = "uint32")
        self.loader.getOutputROIWidth(self.roi_width)
        self.loader.getOutputROIHeight(self.roi_height)
        self.roi_sizes = np.vstack((self.roi_height,self.roi_width)).T # Old
        self.roi_sizes_wh = np.vstack((self.roi_width,self.roi_height)).T # New
#Mask info of a batch
        self.mask_count = np.zeros(self.count_batch, dtype="int32")
        self.mask_size = self.loader.GetMaskCount(self.mask_count)
        self.polygon_size = np.zeros(self.mask_size, dtype= "int32")
        self.mask_data = np.zeros(100000, dtype = "float32")
        self.loader.GetMaskCoordinates(self.polygon_size, self.mask_data)

        count =0
        sum_count=0
        j = 0
        list_poly = []
        iteration1 = 0
        iteration = 0
        self.target_batch = []
        self.roi_image_size = []
        self.roi_image_size_wh = []
        for i in range(self.bs):
            count = self.bboxes_label_count[i]
            self.img_name = self.Img_name[i*16:(i*16)+12]
            self.img_name=self.img_name.decode('utf_8')
            self.img_name = np.char.lstrip(self.img_name, chars ='0')
            self.img_size_2d_numpy = (self.img_size[i*2:(i*2)+2])
            self.img_roi_size2d_numpy = (self.roi_sizes[i])
            self.img_roi_size2d_numpy_wh = (self.roi_sizes_wh[i])
            # Image Size ROI
            roi_tmp_list = self.img_roi_size2d_numpy.tolist()
            self.roi_image_size.append(torch.Size(roi_tmp_list))
            roi_tmp_list = self.img_roi_size2d_numpy_wh.tolist()
            self.roi_image_size_wh.append(torch.Size(roi_tmp_list))
            self.label_2d_numpy = (self.labels[sum_count : sum_count+count])
            self.bb_2d_numpy = (self.bboxes[sum_count*4 : (sum_count+count)*4])
            for index, element in enumerate(self.bb_2d_numpy):
                if index % 2 == 0:
                    self.bb_2d_numpy[index] = self.bb_2d_numpy[index] * self.img_roi_size2d_numpy_wh[0]
                elif index % 2 != 0:
                    self.bb_2d_numpy[index] = self.bb_2d_numpy[index] * self.img_roi_size2d_numpy_wh[1]
            self.bb_2d_numpy = np.reshape(self.bb_2d_numpy, (-1, 4))

            val_dataset = get_val_dataset()
            inv_map = {v: k for k, v in val_dataset.items()}

            self.inv_label = []
            for element in self.label_2d_numpy:
                self.inv_label.append(inv_map[element])

            self.label_2d_numpy = torch.as_tensor(self.inv_label)
            self.iter += 1
            
            self.count_mask = self.bboxes_label_count[i]
            poly_batch_list = []
            for i in range(self.count_mask):
                poly_list = []
                for k in range(self.mask_count[iteration1]):
                    polygons = []
                    polygon_size_check = self.polygon_size[iteration]
                    iteration = iteration + 1
                    for loop_idx in range(polygon_size_check):
                        polygons.append(self.mask_data[j])
                        j = j + 1
                    poly_list.append(polygons)
                iteration1 = iteration1 + 1
                poly_batch_list.append(poly_list)

            self.target_batch.append(self.bb_2d_numpy)
            sum_count = sum_count +count

        for i in range(self.bs):
            img_name = self.Img_name[i*16:(i*16)+12].decode('utf-8')
            image = ((self.out[i].transpose(1,2,0))+[102.9801, 115.9465, 122.7717])[:,:,::-1].astype('uint8')
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            bboxes = self.target_batch[i]
            for box in bboxes:
                x1, y1, x2, y2 = box.astype(np.int)
                cv2.rectangle(image, (x1,y1), (x2,y2), (255, 0, 0), 2)
            cv2.imwrite(f'{self.iter}_{img_name}.jpg', image)
        return self.out


    def reset(self):
        self.loader.raliResetLoaders()

    def __iter__(self):
        self.loader.raliResetLoaders()
        return self
    
def main():
    if len(sys.argv) < 4:
        print('Please pass the arguments image_folder Annotation_file cpu/gpu batch_size')
        exit(0)

    image_path = sys.argv[1]
    ann_path = sys.argv[2]
    if(sys.argv[3] == "cpu"):
        _rali_cpu = True
    else:
        _rali_cpu = False
    bs = int(sys.argv[4])
    random_seed = random.SystemRandom().randint(0, 2**32 - 1)
    
    pipe = Pipeline(batch_size=bs, num_threads=1,device_id=0, seed=random_seed, rali_cpu=_rali_cpu)

    with pipe:
        jpegs, bboxes, labels = fn.readers.coco(
            file_root=image_path, annotations_file=ann_path, random_shuffle=False, seed=random_seed, mask=True)
        images_decoded = fn.decoders.image(jpegs, output_type=types.RGB, file_root=image_path, annotations_file=ann_path, random_shuffle=False, seed=random_seed)
        coin_flip = fn.random.coin_flip(probability=0.5)
        rmn_images = fn.resize_mirror_normalize(images_decoded,
                                            device="gpu",
                                            output_dtype=types.FLOAT16,
                                            output_layout=types.NCHW,
                                            resize_min = 1344,
                                            resize_max = 1344,
                                            mirror=coin_flip,
                                            mean= [102.9801, 115.9465, 122.7717],
                                            std = [1. , 1., 1.])        
        pipe.set_outputs(rmn_images)
    pipe.build()
    data_loader = RALIGenericTrainIterator(
        pipe, reverse_channels=True ,multiplier=pipe._multiplier, offset=pipe._offset)
    epochs = 2
    import timeit
    start = timeit.default_timer()

    for epoch in range(int(epochs)):
        print("EPOCH:::::",epoch)
        for i, it in enumerate(data_loader, 0):
            print("**************", i, "*******************")
            print("**************starts*******************")
            print("**************ends*******************")
            print("**************", i, "*******************")
        data_loader.reset()
    #Your statements here
    stop = timeit.default_timer()

    print('\n Time: ', stop - start)


if __name__ == '__main__':
    main()






