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

class ROCALGenericIterator(object):
    def __init__(self, pipeline, tensor_layout=types.NCHW, reverse_channels=False, multiplier=[1.0, 1.0, 1.0], offset=[0.0, 0.0, 0.0], tensor_dtype=types.FLOAT, device="cpu", device_id=0, display=False, size = -1, auto_reset=False):
        self.loader = pipeline
        self.tensor_format = tensor_layout
        self.multiplier = multiplier
        self.offset = offset
        self.reverse_channels = reverse_channels
        self.tensor_dtype = tensor_dtype
        self.device = device
        self.device_id = device_id
        self.batch_size = self.loader._batch_size
        self.labels_size = ((self.batch_size * self.loader._num_classes) if self.loader._one_hot_encoding else self.batch_size)
        self.output_list = self.dimensions = self.torch_dtype = None
        self.output_memory_type = self.loader._output_memory_type
        self.len = b.getRemainingImages(self.loader._handle)
        self.display = display
        self.last_batch_padded_size = b.getLastBatchPaddedSize(self.loader._handle)
        self.last_batch_policy = self.loader._last_batch_policy
        if self.loader._name is None:
            self.loader._name = self.loader._reader
        self.shard_size = -1#self.loader._shard_size or size
        self.auto_reset = auto_reset
        self.batch_count = 0

    def next(self):
        return self.__next__()

    def __next__(self):
        if self.loader.rocalRun() != 0 and self.shard_size < 0:
            if self.auto_reset:
                self.reset()
            raise StopIteration
        elif self.shard_size > 0 and self.batch_count >= self.shard_size :
            if self.auto_reset:
                self.reset()
            raise StopIteration
        else:
            self.output_tensor_list = self.loader.getOutputTensors()
        self.batch_count += self.batch_size
        self.last_batch_size = self.batch_size - b.getLastBatchPaddedSize(self.loader._handle) #Every Time the padded size is going to differ
        if self.output_list is None:
            self.output_list = []
            for i in range(len(self.output_tensor_list)):
                self.dimensions = self.output_tensor_list[i].dimensions()
                # print("self.dimesnions",self.dimensions)
                if self.device == "cpu":
                    self.torch_dtype = self.output_tensor_list[i].dtype()
                    self.output = torch.empty(self.dimensions, dtype = getattr(torch, self.torch_dtype))
                    self.labels_tensor = torch.empty(self.labels_size, dtype = getattr(torch, self.torch_dtype))
                else:
                    torch_gpu_device = torch.device('cuda', self.device_id)
                    self.torch_dtype = self.output_tensor_list[i].dtype()
                    self.output = torch.empty(self.dimensions, dtype = getattr(torch, self.torch_dtype), device=torch_gpu_device)
                    self.labels_tensor = torch.empty(self.labels_size, dtype = getattr(torch, self.torch_dtype), device=torch_gpu_device)

                self.output_tensor_list[i].copy_data(ctypes.c_void_p(self.output.data_ptr()), self.output_memory_type)
                self.output_list.append(self.output)
        else:
            for i in range(len(self.output_tensor_list)):
                self.output_tensor_list[i].copy_data(ctypes.c_void_p(self.output_list[i].data_ptr()), self.output_memory_type)

        if ((self.loader._name == "Caffe2ReaderDetection") or (self.loader._name == "CaffeReaderDetection")):
            self.bbox_list = []  # Empty list for bboxes
            self.labels_list = []  # Empty list of labels

            # 1D labels array in a batch
            self.labels = self.loader.getBoundingBoxLabels()
            # 1D bboxes array in a batch
            self.bboxes = self.loader.rocalGetBoundingBoxCords()
            # Image sizes of a batch
            self.img_size = np.zeros((self.batch_size * 2), dtype="int32")
            self.loader.getImgSizes(self.img_size)

            for i in range(self.batch_size):

                self.label_2d_numpy = (self.labels[i])
                self.label_2d_numpy = np.reshape(self.label_2d_numpy, (-1, 1)).tolist()
                self.bb_2d_numpy = (self.bboxes[i])
                self.bb_2d_numpy = np.reshape(self.bb_2d_numpy, (-1, 4)).tolist()

                self.labels_list.append(self.label_2d_numpy)
                self.bbox_list.append(self.bb_2d_numpy)

                if self.display:
                    for output in self.output_list:
                        img = (output)
                        draw_patches(img[i], i, self.bb_2d_numpy)


            max_cols = max([len(row) for batch in self.bbox_list for row in batch])
            max_rows = max([len(batch) for batch in self.bbox_list])
            self.bb_padded = [batch + [[0] * (max_cols)] * (max_rows - len(batch)) for batch in self.bbox_list]
            self.bb_padded = torch.FloatTensor([row + [0] * (max_cols - len(row)) for batch in self.bb_padded for row in batch])
            self.bb_padded = self.bb_padded.view(-1, max_rows, max_cols)

            max_cols1 = max([len(row) for batch in self.labels_list for row in batch])
            max_rows1 = max([len(batch) for batch in self.labels_list])
            self.labels_padded = [batch + [[0] * (max_cols1)] * (max_rows1 - len(batch)) for batch in self.labels_list]
            self.labels_padded = torch.LongTensor([row + [0] * (max_cols1 - len(row)) for batch in self.labels_padded for row in batch])
            self.labels_padded = self.labels_padded.view(-1, max_rows1, max_cols1)
            if (self.last_batch_policy is (types.LAST_BATCH_PARTIAL)) and b.getRemainingImages(self.loader._handle) == 0 and b.getLastBatchPaddedSize(self.loader._handle) > 0: #Check this condition
                return [inner_list[0:self.last_batch_size,:] for inner_list in self.output_list], self.bb_padded[0:self.last_batch_size], self.labels_padded[0:self.last_batch_size]
            else:
                return self.output_list, self.bb_padded, self.labels_padded

        else:
            if self.loader._one_hot_encoding:
                self.loader.getOneHotEncodedLabels(self.labels_tensor, self.device)
                self.labels_tensor = self.labels_tensor.reshape(-1, self.batch_size, self.loader._num_classes)
            else:
                if self.display:
                    for i in range(self.batch_size):
                        img = (self.output)
                        draw_patches(img[i], i, 0)
                self.labels = self.loader.getImageLabels()
                self.labels_tensor = self.labels_tensor.copy_(torch.from_numpy(self.labels)).long()
            if (self.last_batch_policy is (types.LAST_BATCH_PARTIAL)) and b.getRemainingImages(self.loader._handle) == 0 and b.getLastBatchPaddedSize(self.loader._handle) > 0: #Check this condition
                return [inner_list[0:self.last_batch_size,:] for inner_list in self.output_list], self.labels_tensor[0:self.last_batch_size]
            else:
                return self.output_list, self.labels_tensor

    def reset(self):
        self.batch_count = 0
        b.rocalResetLoaders(self.loader._handle)

    def __iter__(self):
        return self

    def __len__(self):
        return self.len

    def __del__(self):
        print("In rocALRelease")
        b.rocalRelease(self.loader._handle)


class ROCALClassificationIterator(ROCALGenericIterator):
    """
    ROCAL iterator for classification tasks for PyTorch. It returns 2 outputs
    (data and label) in the form of PyTorch's Tensor.

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
                 change during execution. If True, the pytorch tensor will be resized accordingly
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
                 size=-1,
                 auto_reset=False,
                 fill_last_batch=True,
                 dynamic_shape=False,
                 last_batch_padded=False,
                 display=False,
                 device="cpu",
                 device_id=0,):
        pipe = pipelines
        super(ROCALClassificationIterator, self).__init__(pipe, tensor_layout=pipe._tensor_layout, tensor_dtype=pipe._tensor_dtype,
                                                          multiplier=pipe._multiplier, offset=pipe._offset, display=display, device=device, device_id=device_id, size = size, auto_reset = auto_reset)


def draw_patches(img, idx, bboxes):
    # image is expected as a tensor
    import cv2
    img = img.cpu()
    image = img.detach().numpy()
    image = image.transpose([1, 2, 0])
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.UMat(image).get()
    cv2.imwrite(str(idx)+"_"+"train"+".png", image)
    try:
        path = "OUTPUT_IMAGES_PYTHON/PYTORCH/"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
    except OSError as error:
        print(error)
    if bboxes:
        for (l, t, r, b) in bboxes:
            loc_ = [l, t, r, b]
            color = (255, 0, 0)
            thickness = 2
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = cv2.UMat(image).get()
            image = cv2.rectangle(image, (int(loc_[0]), int(loc_[1])), (int(
                (loc_[2])), int((loc_[3]))), color, thickness)
            cv2.imwrite("OUTPUT_IMAGES_PYTHON/PYTORCH/" + str(idx) + "_" + "train" + ".png", image * 255)
    else:
        cv2.imwrite("OUTPUT_IMAGES_PYTHON/PYTORCH/" + str(idx) + "_" + "train" + ".png", image * 255)

class ROCALAudioIterator(object):
    """
    ROCAL iterator for audio tasks for PyTorch

    Please keep in mind that Tensors returned by the iterator are
    still owned by ROCAL. They are valid till the next iterator call.
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
    def __init__(self, pipeline, tensor_layout = types.NCHW, reverse_channels = False, multiplier = [1.0, 1.0, 1.0], offset = [0.0, 0.0, 0.0], tensor_dtype = types.FLOAT, size = -1, auto_reset = False, device = "cpu", device_id = 0):
        self.loader = pipeline
        self.tensor_format = tensor_layout
        self.multiplier = multiplier
        self.offset = offset
        self.reverse_channels = reverse_channels
        self.tensor_dtype = tensor_dtype
        self.device = device
        self.device_id = device_id
        self.output = None
        self.len = b.getRemainingImages(self.loader._handle)
        # self.last_batch_policy = self.loader._last_batch_policy #commented for now
        self.shard_size = size
        self.auto_reset = auto_reset
        self.batch_count = 0
        self.audio_length = None
        self.samples = None
        self.channels = None
        self.max_shape = None
        self.batch_size = self.loader._batch_size

    def next(self):
        return self.__next__()

    def __next__(self):
        if self.loader.rocalRun() != 0:
            raise StopIteration
        else:
            self.output_tensor_list = self.loader.getOutputTensors()

        self.batch_count = self.batch_count + self.batch_size
        self.num_of_dims = self.output_tensor_list[0].num_of_dims()
        if self.num_of_dims == 3:
            self.batch_size = self.output_tensor_list[0].batch_size() if self.batch_size is None else self.batch_size
            self.max_shape = self.output_tensor_list[0].max_shape() if self.max_shape is None else self.max_shape
            self.channels = self.max_shape[0]
            self.samples = self.max_shape[1]
            self.audio_length = self.channels * self.samples if self.audio_length is None else self.audio_length
            roi = self.output_tensor_list[0].get_rois().reshape(self.batch_size, 4)
            max_x1 = np.max(roi[..., 0:1])
            max_y1 = np.max(roi[..., 1:2])
            self.output = torch.empty((self.batch_size, max_y1, max_x1), dtype=torch.float32)

            self.labels = self.loader.getImageLabels()
            self.labels_tensor = torch.from_numpy(self.labels).type(torch.LongTensor)

            self.output_tensor_list[0].copy_data(ctypes.c_void_p(self.output.data_ptr()), max_y1, max_x1)
            return self.output, self.labels_tensor, torch.tensor(roi[..., 0:2])

    def reset(self):
        b.rocalResetLoaders(self.loader._handle)

    def __iter__(self):
        return self

    def __len__(self):
        return self.len

    def __del__(self):
        print("Comes to rocALRelease")
        b.rocalRelease(self.loader._handle)