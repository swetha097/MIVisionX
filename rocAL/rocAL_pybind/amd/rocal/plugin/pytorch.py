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
import numpy as np
import rocal_pybind as b
import amd.rocal.types as types
import ctypes

# class RALIGenericImageIterator(object):
#     def __init__(self, pipeline):
#         self.loader = pipeline
#         self.w = b.getOutputWidth(self.loader._handle)
#         self.h = b.getOutputHeight(self.loader._handle)
#         self.n = b.getOutputImageCount(self.loader._handle)
#         color_format = b.getOutputColorFormat(self.loader._handle)
#         self.p = (1 if (color_format == int(types.GRAY)) else 3)
#         height = self.h*self.n
#         self.out_tensor = None
#         self.out_bbox = None
#         self.out_image = np.zeros((height, self.w, self.p), dtype = "uint8")
#         self.bs = pipeline._batch_size

#     def next(self):
#         return self.__next__()

#     def __next__(self):
#         if b.getRemainingImages(self.loader._handle) < self.bs:
#             raise StopIteration

#         if self.loader.run() != 0:
#             raise StopIteration

#         self.loader.copyImage(self.out_image)
#         if((self.loader._name == "Caffe2ReaderDetection") or (self.loader._name == "CaffeReaderDetection")):

#             for i in range(self.bs):
#                 size = b.getImageNameLen(self.loader._handle,i)
#                 print(size)
#                 self.array = np.array(["                 "])

#                 self.out=np.frombuffer(self.array, dtype=(self.array).dtype)

#                 b.getImageName(self.loader._handle, self.out ,i)
#             return self.out_image ,self.out_bbox, self.out_tensor
#         else:
#             return self.out_image , self.out_tensor

#     def reset(self):
#         b.raliResetLoaders(self.loader._handle)

#     def __iter__(self):
#         return self


class ROCALGenericIterator(object):
    def __init__(self, pipeline, tensor_layout = types.NCHW, reverse_channels = False, multiplier = [1.0,1.0,1.0], offset = [0.0, 0.0, 0.0], tensor_dtype=types.FLOAT, device="cpu", device_id=0):
        self.loader = pipeline
        self.tensor_format =tensor_layout
        self.multiplier = multiplier
        self.offset = offset
        self.reverse_channels = reverse_channels
        self.tensor_dtype = tensor_dtype
        self.len = b.getRemainingImages(self.loader._handle)
        self.last_batch_padded_size = b.getLastBatchPaddedSize(self.loader._handle)
        self.last_batch_policy = self.loader._last_batch_policy
        self.shard_size = size
        self.auto_reset = auto_reset
        self.batch_count = 0
        self.batch_size = None
        self.audio_length = None
        self.samples = None
        self.channels = None
        self.output = None
        self.batch_size = self.loader._batch_size

    def next(self):
        return self.__next__()

    def __next__(self):
        if(b.isEmpty(self.loader._handle)) and self.shard_size < 0:
            print("Handle Empty")
            if self.auto_reset:
                self.reset()
            raise StopIteration

        if (self.loader.rocalRun() != 0 and self.shard_size < 0):
            print("rocALRun() ! =0")
            if self.auto_reset:
                self.reset()
            raise StopIteration

        elif self.shard_size > 0 and self.batch_count >= self.shard_size :
            if self.auto_reset:
                self.reset()
            raise StopIteration

        else:
            self.output_tensor_list = self.loader.rocalGetOutputTensors()

        self.last_batch_padded_size = b.getLastBatchPaddedSize(self.loader._handle)
        self.last_batch_size = self.batch_size - self.last_batch_padded_size
        self.batch_count+=self.batch_size
        #From init
        self.num_of_dims = self.output_tensor_list[0].num_of_dims()
        if self.num_of_dims == 4: # In the case of the Image data
            self.w = self.output_tensor_list[0].batch_width()
            self.h = self.output_tensor_list[0].batch_height()
            self.batch_size = self.output_tensor_list[0].batch_size()
            self.color_format = self.output_tensor_list[0].color_format()

            if self.out is None:
                if self.tensor_format == types.NCHW:
                    torch_gpu_device = torch.device('cuda', self.device_id)
                    if self.tensor_dtype == types.FLOAT:
                        self.out = torch.empty((self.batch_size, self.color_format, self.h, self.w,), dtype=torch.float32, device = torch_gpu_device)
                    elif self.tensor_dtype == types.FLOAT16:
                        self.out = torch.empty((self.batch_size, self.color_format, self.h, self.w,), dtype=torch.float16, device = torch_gpu_device)                

                else: #NHWC
                    torch_gpu_device = torch.device('cuda', self.device_id)
                    if self.tensor_dtype == types.FLOAT:
                        self.out = torch.empty((self.batch_size, self.h, self.w, self.color_format), dtype=torch.float32, device=torch_gpu_device)
                    elif self.tensor_dtype == types.FLOAT16:
                        self.out = torch.empty((self.batch_size, self.h, self.w, self.color_format), dtype=torch.float16, device=torch_gpu_device)
                
                self.labels_tensor = torch.empty(self.batch_size, dtype = torch.int32, device = torch_gpu_device)

            self.output_tensor_list[0].copy_data(ctypes.c_void_p(self.out.data_ptr()))
            self.labels = self.loader.rocalGetImageLabels()
            self.labels_tensor = self.labels_tensor.copy_(torch.from_numpy(self.labels)).long()
            if self.tensor_dtype == types.FLOAT:
                return self.out, self.labels_tensor
            elif self.tensor_dtype == types.FLOAT16:
                return self.out.half(), self.labels_tensor
        elif self.num_of_dims == 3: #In case of an audio data
            self.batch_size = self.output_tensor_list[0].batch_size() if self.batch_size is None else self.batch_size
            self.channels = self.output_tensor_list[0].batch_width() if self.channels is None else self.channels #Max Channels
            self.samples = self.output_tensor_list[0].batch_height() if self.samples is None else self.samples #Max Samples
            self.audio_length = self.channels * self.samples if self.audio_length is None else self.audio_length
            roi = self.output_tensor_list[0].get_rois().reshape(self.batch_size,4)
            x1 = torch.tensor(roi[...,0:1])
            y1 = torch.tensor(roi[...,1:2])
            max_x1 = torch.max(x1)
            max_y1 = torch.max(y1)
            self.output = torch.empty((self.batch_size, max_y1, max_x1,), dtype=torch.float32)
            # next
            self.labels = self.loader.rocalGetImageLabels()
            self.labels_tensor = torch.from_numpy(self.labels).type(torch.LongTensor)
            if (self.last_batch_policy is (types.LAST_BATCH_PARTIAL)) and b.getRemainingImages(self.loader._handle) <= 0 :
                self.output_tensor_list[0].copy_data(ctypes.c_void_p(self.output.data_ptr()), max_y1, max_x1)
                return self.output[0:self.last_batch_size,:], self.labels_tensor[0:self.last_batch_size], torch.tensor(self.output_tensor_list[0].get_rois().reshape(self.batch_size,4)[...,0:2][0:self.last_batch_size,:])
            else:
                self.output_tensor_list[0].copy_data(ctypes.c_void_p(self.output.data_ptr()), max_y1, max_x1)
                return self.output, self.labels_tensor, torch.tensor(self.output_tensor_list[0].get_rois().reshape(self.batch_size,4)[...,0:2])

    def reset(self):
        self.batch_count = 0
        b.rocalResetLoaders(self.loader._handle)

    def __iter__(self):
        return self

    def __len__(self):
        return self.len

    def __del__(self):
        b.rocalRelease(self.loader._handle)


class ROCALClassificationIterator(ROCALGenericIterator):
    """
    RALI iterator for classification tasks for PyTorch. It returns 2 outputs
    (data and label) in the form of PyTorch's Tensor.

    Calling

    .. code-block:: python

       ROCALClassificationIterator(pipelines, size)

    is equivalent to calling

    .. code-block:: python

       ROCALGenericIterator(pipelines, ["data", "label"], size)

    Please keep in mind that Tensors returned by the iterator are
    still owned by RALI. They are valid till the next iterator call.
    If the content needs to be preserved please copy it to another tensor.

    Parameters
    ----------
    pipelines : list of amd.rocalLI.pipeline.Pipeline
                List of pipelines to use
    size : int
           Number of samples in the epoch (Usually the size of the dataset).
    auto_reset : bool, optional, default = False
                 Whether the iterator resets itself for the next epoch
                 or it requires reset() to be called separately.
    fill_last_batch : bool, optional, default = True
                 Whether to fill the last batch with data up to 'self.batch_size'.
                 The iterator would return the first integer multiple
                 of self._num_gpus * self.batch_size entries which exceeds 'size'.
                 Setting this flag to False will cause the iterator to return
                 exactly 'size' entries.
    dynamic_shape: bool, optional, default = False
                 Whether the shape of the output of the RALI pipeline can
                 change during execution. If True, the pytorch tensor will be resized accordingly
                 if the shape of RALI returned tensors changes during execution.
                 If False, the iterator will fail in case of change.
    last_batch_padded : bool, optional, default = False
                 Whether the last batch provided by RALI is padded with the last sample
                 or it just wraps up. In the conjunction with `fill_last_batch` it tells
                 if the iterator returning last batch with data only partially filled with
                 data from the current epoch is dropping padding samples or samples from
                 the next epoch. If set to False next epoch will end sooner as data from
                 it was consumed but dropped. If set to True next epoch would be the
                 same length as the first one.

    Example
    -------
    With the data set [1,2,3,4,5,6,7] and the batch size 2:
    fill_last_batch = False, last_batch_padded = True  -> last batch = [7], next iteration will return [1, 2]
    fill_last_batch = False, last_batch_padded = False -> last batch = [7], next iteration will return [2, 3]
    fill_last_batch = True, last_batch_padded = True   -> last batch = [7, 7], next iteration will return [1, 2]
    fill_last_batch = True, last_batch_padded = False  -> last batch = [7, 1], next iteration will return [2, 3]
    """
    def __init__(self,
                 pipelines,
                 device="cpu",
                 device_id=0,
                 size = 0,
                 auto_reset=False,
                 fill_last_batch=True,
                 dynamic_shape=False,
                 last_batch_padded=False):
        pipe = pipelines
        super(ROCALClassificationIterator, self).__init__(pipe, tensor_layout = pipe._tensor_layout, tensor_dtype = pipe._tensor_dtype,
                                                            multiplier=pipe._multiplier, offset=pipe._offset, device=device, device_id=device_id)


# class RALI_iterator(RALIGenericImageIterator):
#     """
#     RALI iterator for classification tasks for PyTorch. It returns 2 outputs
#     (data and label) in the form of PyTorch's Tensor.

#     """
#     def __init__(self,
#                  pipelines,
#                  size = 0,
#                  auto_reset=False,
#                  fill_last_batch=True,
#                  dynamic_shape=False,
#                  last_batch_padded=False):
#         pipe = pipelines
#         super(RALI_iterator, self).__init__(pipe)
