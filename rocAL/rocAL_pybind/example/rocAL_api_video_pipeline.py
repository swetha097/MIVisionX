from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import amd.rocal.types as types
import numpy as np
from parse_config import parse_args
import sys
import cv2
import os
from math import sqrt
import ctypes
import torch


class ROCALVideoIterator(object):
    """
    ROCALVideoIterator for pyTorch.

    Parameters
    ----------
    pipelines : list of amd.rocal.pipeline.Pipeline
                List of pipelines to use
    size : int
           Epoch size.
    """

    def __init__(self, pipelines, tensor_layout=types.NFHWC, reverse_channels=False, multiplier=None, offset=None, tensor_dtype=types.FLOAT, device="cpu", display=False ,sequence_length=3):

        try:
            assert pipelines is not None, "Number of provided pipelines has to be at least 1"
        except Exception as ex:
            print(ex)
        self.loader = pipelines
        self.tensor_format = tensor_layout
        print("Tensor layout : ", self.tensor_format)
        self.multiplier = multiplier if multiplier else [1.0, 1.0, 1.0]
        self.offset = offset if offset else [0.0, 0.0, 0.0]
        self.reverse_channels = reverse_channels
        self.tensor_dtype = tensor_dtype
        print("\nself.tensor_dtype : ", self.tensor_dtype)
        self.device = device
        self.device_id = self.loader._device_id
        self.rim = self.loader.getRemainingImages()
        self.display = display
        self.iter_num = 0
        self.sequence_length = sequence_length
        print("____________REMAINING IMAGES____________:", self.rim)

    def next(self):
        return self.__next__()

    def __next__(self):
        self.iter_num +=1
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
        self.p = (1 if self.color_format is types.GRAY else 3)
        #NFHWC format default for now
        self.out = torch.empty((self.bs, self.sequence_length, self.h, self.w, self.p), dtype=torch.uint8)
        self.output_tensor_list[0].copy_data(ctypes.c_void_p(self.out.data_ptr()))
        print("\n Images : ", self.out)
        #print("\n Imag 0 : ", self.out[0][0])
        #Display Frames in a video sequence
        #if self.display:
        for batch_i in range(self.bs):
            for seq in range(self.sequence_length):
                draw_frames(self.out[batch_i][seq], batch_i, self.iter_num, self.device)
        return self.out
 
    def reset(self):
        self.loader.rocalResetLoaders()

    def __iter__(self):
        return self

def draw_frames(img,batch_idx,iter_idx, device):
    #image is expected as a tensor, bboxes as numpy
    import cv2
    if device == "cpu":
        image = img.detach().numpy()
    else:
        image = img.cpu().numpy()
    #print('Shape is:',image.shape)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR )
    import os
    if not os.path.exists("OUTPUT_IMAGES_PYTHON/NEW_API/VIDEO_READER"):
        os.makedirs("OUTPUT_IMAGES_PYTHON/NEW_API/VIDEO_READER")
    image = cv2.UMat(image).get()
    cv2.imwrite("OUTPUT_IMAGES_PYTHON/NEW_API/VIDEO_READER/"+"iter_"+str(iter_idx)+"_batch_"+str(batch_idx)+".png", image)

def main():
    #Args
    args = parse_args()
    video_path = args.video_path
    _rocal_cpu = False if args.rocal_gpu else True
    batch_size = args.batch_size
    user_sequence_length = args.sequence_length
    display = args.display
    num_threads = args.num_threads
    random_seed = args.seed
    # Create Pipeline instance
    pipe = Pipeline(batch_size=batch_size, num_threads=num_threads,device_id=args.local_rank, seed=random_seed, rocal_cpu=_rocal_cpu)
    # Use pipeline instance to make calls to reader, decoder & augmentation's
    with pipe:
        images = fn.readers.video(device="gpu", file_root=video_path, sequence_length=user_sequence_length,
                              normalized=False, random_shuffle=False, image_type=types.RGB,
                              dtype=types.UINT8, initial_fill=16, pad_last_batch=True, name="Reader")
        crop_size = (512,960)
        output_images = fn.crop_mirror_normalize(images,
                                            crop=crop_size,
                                            mean=[0, 0, 0],
                                            std=[1,1,1],
                                            #mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            #std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                                            mirror=0,
                                            rocal_tensor_output_type=types.FLOAT,
                                            rocal_tensor_layout=types.NFHWC,
                                            pad_output=False)
        pipe.set_outputs(images)
    # Build the pipeline
    pipe.build()
    # Dataloader
    #data_loader = ROCALVideoIterator(
       #pipe, multiplier=pipe._multiplier, offset=pipe._offset, device = self.device,display=display,sequence_length=user_sequence_length)
    data_loader = ROCALVideoIterator(pipe)
    import timeit
    start = timeit.default_timer()
    # Enumerate over the Dataloader
    for epoch in range(int(args.num_epochs)):
        print("EPOCH:::::",epoch)
        for i, it in enumerate(data_loader, 0):
            if args.print_tensor:
                print("**************", i, "*******************")
                print("**************starts*******************")
                print("\n IMAGES : \n", it)
                print("**************ends*******************")
                print("**************", i, "*******************")
        data_loader.reset()
    #Your statements here
    stop = timeit.default_timer()

    print('\n Time: ', stop - start)


if __name__ == '__main__':
    main()






