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

class RALIVideoIterator(object):
    """
    COCO RALI iterator for pyTorch.

    Parameters
    ----------
    pipelines : list of amd.rali.pipeline.Pipeline
                List of pipelines to use
    size : int
           Epoch size.
    """

    def __init__(self, pipelines, tensor_layout=types.NCHW, reverse_channels=False, multiplier=None, offset=None, tensor_dtype=types.FLOAT, display=False ,sequence_length=3):

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
        self.iter_num = 0
        self.sequence_length = sequence_length
        print("____________REMAINING IMAGES____________:", self.rim)
        color_format = self.loader.getOutputColorFormat()
        self.p = (1 if color_format is types.GRAY else 3)
        self.out = np.empty(
                (self.bs*self.n,int(self.h/self.bs), self.w,self.p), dtype="ubyte")
        
    def next(self):
        return self.__next__()

    def __next__(self):
        self.iter_num +=1
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

        #Copy output from buffer to numpy array
        self.loader.copyImage(self.out)
        img = torch.from_numpy(self.out)

        #Display Frames in a video sequence
        if self.display:
            for batch_i in range(self.bs):
                draw_frames(img[batch_i], batch_i, self.iter_num)


        return img

    def reset(self):
        self.loader.raliResetLoaders()

    def __iter__(self):
        return self

def draw_frames(img,batch_idx,iter_idx):
    #image is expected as a tensor, bboxes as numpy
    import cv2
    image = img.detach().numpy()
    # print('Shape is:',img.shape)
    image = image.transpose([0,1,2])
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR )
    #mkdir a dir to dump the video frames
    import os
    if not os.path.exists("output_video_frames_dump"):
        os.makedirs("output_video_frames_dump")
    _,htot ,wtot = img.shape
    image = cv2.UMat(image).get()
    cv2.imwrite("output_video_frames_dump/"+"iter_"+str(iter_idx)+"_batch_"+str(batch_idx)+".png", image)

def main():
    if len(sys.argv) < 5:
        print('Please pass the folder image_folder cpu/gpu batch_size sequence_length display(True/False)')
        exit(0)

    video_path = sys.argv[1]
    if(sys.argv[2] == "cpu"):
        _rali_cpu = True
    else:
        _rali_cpu = False
    bs = int(sys.argv[3])
    user_sequence_length = int(sys.argv[4])
    
    display = sys.argv[5]
    nt = 1
    di = 0
    crop_size = 300
    random_seed = random.SystemRandom().randint(0, 2**32 - 1)
    pipe = Pipeline(batch_size=bs, num_threads=1,device_id=0, seed=random_seed, rali_cpu=_rali_cpu)

    with pipe:
        images = fn.readers.video(device="gpu", file_root=video_path, sequence_length=user_sequence_length,
                              normalized=False, random_shuffle=False, image_type=types.RGB,
                              dtype=types.FLOAT, initial_fill=16, pad_last_batch=True, name="Reader")
        crop_size = (512,960)
        '''
        output_images = fn.crop(images, crop=crop_size,
                         crop_pos_x=fn.random.uniform(range=(0.0, 1.0)),
                         crop_pos_y=fn.random.uniform(range=(0.0, 1.0)))
        '''
        output_images = fn.crop_mirror_normalize(images,
                                            crop=crop_size,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                                            mirror=0,
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            pad_output=False)
        pipe.set_outputs(output_images)
    pipe.build()

    data_loader = RALIVideoIterator(
        pipe, multiplier=pipe._multiplier, offset=pipe._offset,display=display,sequence_length=user_sequence_length)
    epochs = 1
    for epoch in range(int(epochs)):
        print("EPOCH:::::",epoch)
        for i, it in enumerate(data_loader, 0):
            print("**************", i, "*******************")
            print("**************starts*******************")
            print("\n IMAGES : \n", it)
            print("**************ends*******************")
            print("**************", i, "*******************")
        data_loader.reset()


if __name__ == '__main__':
    main()


    



