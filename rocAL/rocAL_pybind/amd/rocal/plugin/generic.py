# Copyright (c) 2018 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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

import cupy as cp
import numpy as np
import rocal_pybind as b
import amd.rocal.types as types


class ROCALGenericIterator(object):
    def __init__(self, pipeline, tensor_layout=types.NCHW, reverse_channels=False, multiplier=[1.0, 1.0, 1.0], offset=[0.0, 0.0, 0.0], tensor_dtype=types.FLOAT, display=False, device="cpu", device_id=0):
        self.loader = pipeline
        self.tensor_format = tensor_layout
        self.multiplier = multiplier
        self.offset = offset
        self.device = device
        self.device_id = device_id
        self.reverse_channels = reverse_channels
        self.tensor_dtype = tensor_dtype
        self.display = display
        self.w = b.getOutputWidth(self.loader._handle)
        self.h = b.getOutputHeight(self.loader._handle)
        self.n = b.getOutputImageCount(self.loader._handle)
        self.bs = pipeline._batch_size
        self.index = 0
        self.eos = False
        if self.loader._name is None:
            self.loader._name= self.loader._reader
        color_format = b.getOutputColorFormat(self.loader._handle)
        self.p = (1 if (color_format == int(types.GRAY)) else 3)
        self.labels_size = ((self.bs*self.loader._numOfClasses) if (self.loader._oneHotEncoding == True) else self.bs)
        if tensor_layout == types.NCHW:
            if self.device == "cpu":
                if self.tensor_dtype == types.FLOAT:
                    self.out = np.empty((self.bs*self.n, self.p, int(self.h/self.bs), self.w,), dtype=np.float32)
                elif self.tensor_dtype == types.FLOAT16:
                    self.out = np.empty((self.bs*self.n, self.p, int(self.h/self.bs), self.w,), dtype=np.float16)
                self.labels = np.empty(self.labels_size, dtype = np.int32)

            else:
                with cp.cuda.Device(device=self.device_id):
                    if self.tensor_dtype == types.FLOAT:
                        self.out = cp.empty((self.bs*self.n, self.p, int(self.h/self.bs), self.w,), dtype=cp.float32)
                    elif self.tensor_dtype == types.FLOAT16:
                        self.out = cp.empty((self.bs*self.n, self.p, int(self.h/self.bs), self.w,), dtype=cp.float16)
                    self.labels = cp.empty(self.labels_size, dtype = cp.int32)

        else: #NHWC
            if self.device == "cpu":
                if self.tensor_dtype == types.FLOAT:
                    self.out = np.empty((self.bs*self.n, int(self.h/self.bs), self.w, self.p), dtype=np.float32)
                elif self.tensor_dtype == types.FLOAT16:
                    self.out = np.empty((self.bs*self.n, int(self.h/self.bs), self.w, self.p), dtype=np.float16)
                self.labels = np.empty(self.labels_size, dtype = np.int32)

            else:
                with cp.cuda.Device(device=self.device_id):
                    if self.tensor_dtype == types.FLOAT:
                        self.out = cp.empty((self.bs*self.n, int(self.h/self.bs), self.w, self.p), dtype=cp.float32)
                    elif self.tensor_dtype == types.FLOAT16:
                        self.out = cp.empty((self.bs*self.n, int(self.h/self.bs), self.w, self.p), dtype=cp.float16)
                    self.labels = cp.empty(self.labels_size, dtype = cp.int32)


        if self.bs != 0:
            self.len = b.getRemainingImages(self.loader._handle)//self.bs
        else:
        self.len = b.getRemainingImages(self.loader._handle)
        self.num_batches = self.loader._external_source.n // self.bs if self.loader._external_source.n % self.bs == 0 else (self.loader._external_source.n // self.bs + 1)

    def next(self):
        return self.__next__()

    def __next__(self):
        if self.loader.rocal_run() != 0:
            raise StopIteration
        self.output_tensor_list = self.loader.get_output_tensors()

        if self.output_list is None:  # Checking if output_list is empty and initializing the buffers
            self.output_list = []
            for i in range(len(self.output_tensor_list)):
                self.dimensions = self.output_tensor_list[i].dimensions()
                if self.device == "cpu":
                    self.dtype = self.output_tensor_list[i].dtype()
                    self.output = np.empty(self.dimensions, dtype=self.dtype)
                    self.labels = np.empty(self.labels_size, dtype=self.dtype)
                else:
                    self.dtype = self.output_tensor_list[i].dtype()
                    with cp.cuda.Device(device=self.device_id):
                        self.output = cp.empty(self.dimensions, dtype=self.dtype)
                        self.labels = cp.empty(self.labels_size, dtype=self.dtype)

        if (self.loader._external_source_operator):
            if (self.index + 1) == self.num_batches:
                self.eos = True
            if (self.index + 1) <= self.num_batches:
                if self.loader._external_source_mode == types.EXTSOURCE_FNAME:
                    kwargs_pybind = {
                        "handle":self.loader._handle,
                        "source_input_images":next(self.loader._external_source)[0],
                        "labels":next(self.loader._external_source)[1],
                        "input_batch_buffer":[],
                        "roi_width":[],
                        "roi_height":[],
                        "decoded_width":self.loader._external_source_user_given_width,
                        "decoded_height":self.loader._external_source_user_given_height,
                        "channels":self.p,
                        "external_source_mode":self.loader._external_source_mode,
                        "rocal_tensor_layout":types.NCHW,
                        "eos":self.eos }
                    b.ExternalSourceFeedInput(*(kwargs_pybind.values()))
                if self.loader._external_source_mode == types.EXTSOURCE_RAW_COMPRESSED:
                    print("Support for EXTSOURCE_RAW_COMPRESSED / Mode 1 does not exist ")
                    exit(0)
                if self.loader._external_source_mode == types.EXTSOURCE_RAW_UNCOMPRESSED:
                    print("Support for EXTSOURCE_RAW_UNCOMPRESSED / Mode 2 does not exist ")
                    exit(0)

        if(types.NCHW == self.tensor_format):
            self.loader.copyToExternalTensorNCHW(self.out, self.multiplier, self.offset, self.reverse_channels, int(self.tensor_dtype))
        else:
            for i in range(len(self.output_tensor_list)):
                if self.device == "cpu":
                    self.output_tensor_list[i].copy_data(self.output_list[i])
                else:
                    self.output_tensor_list[i].copy_data(self.output_list[i].data.ptr)

        if self.loader._name == "labelReader":
            if self.loader._one_hot_encoding == True:
                self.loader.get_one_hot_encoded_labels(self.labels, self.device)
                self.labels_tensor = self.labels.reshape(-1, self.batch_size, self.loader._num_classes)
            else:
                if self.display:
                    for output in self.output_tensor_list:
                        for i in range(self.batch_size):
                            draw_patches(output[i], i)
                self.labels = self.loader.get_image_labels()
                if self.device == "cpu":
                    self.labels_tensor = self.labels.astype(dtype=np.int_)
                else:
                    with cp.cuda.Device(device=self.device_id):
                        self.labels_tensor = self.labels.astype(dtype=cp.int_)

        return self.output_list, self.labels_tensor

    def reset(self):
        b.rocalResetLoaders(self.loader._handle)

    def __iter__(self):
        return self

    def __len__(self):
        return self.iterator_length

    def __del__(self):
        b.rocalRelease(self.loader._handle)


class ROCALClassificationIterator(ROCALGenericIterator):
    """
    ROCAL iterator for classification tasks for generic images. It returns 2 outputs
    (data and label) in the form of numpy/cupy Tensor.

    Calling

    .. code-block:: python

       ROCALClassificationIterator(pipelines, size)

    is equivalent to calling

    .. code-block:: python

       ROCALGenericIterator(pipelines, ["data", "label"], size)

    Please keep in mind that Tensors returned by the iterator are
    still owned by ROCAL. They are valid till the next iterator call.
    If the content needs to be preserved please copy it to another tensor.

    Parameters
    ----------
    pipelines : list of amd.rocal.pipeline.Pipeline
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
                 Whether the shape of the output of the ROCAL pipeline can
                 change during execution. If True, the numpy tensor will be resized accordingly
                 if the shape of ROCAL returned tensors changes during execution.
                 If False, the iterator will fail in case of change.
    last_batch_padded : bool, optional, default = False
                 Whether the last batch provided by ROCAL is padded with the last sample
                 or it just wraps up. In the conjunction with `fill_last_batch` it tells
                 if the iterator returning last batch with data only partially filled with
                 data from the current epoch is dropping padding samples or samples from
                 the next epoch. If set to False next epoch will end sooner as data from
                 it was consumed but dropped. If set to True next epoch would be the
                 same length as the first one.
    display : bool, optional, default = False
                 Whether the images should be saved as png files for display.
    device : str, optional, default = "cpu"
                 Whether to use CPU or GPU backend for the dataloader.
    device_id : int, optional, default = 0
                 Device ID of the GPU being used for the training loader.

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
                 size=0,
                 auto_reset=False,
                 fill_last_batch=True,
                 dynamic_shape=False,
                 last_batch_padded=False,
                 display=False,
                 device="cpu",
                 device_id=0):
        pipe = pipelines
        super(ROCALClassificationIterator, self).__init__(pipe, tensor_layout=pipe._tensor_layout, tensor_dtype=pipe._tensor_dtype,
                                                          multiplier=pipe._multiplier, offset=pipe._offset, display=display, device=device, device_id=device_id)


def draw_patches(img, idx):
    # image is expected as an array
    import cv2
    img = img.cpu()
    image = img.detach().numpy()
    image = image.transpose([1, 2, 0])
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.UMat(image).get()
    cv2.imwrite(str(idx)+"_"+"train"+".png", image)
