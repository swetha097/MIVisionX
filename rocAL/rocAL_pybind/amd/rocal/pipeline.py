import rocal_pybind as b
import amd.rocal.types as types
import numpy as np
import torch
import ctypes


class Pipeline(object):

    """Pipeline class internally calls RocalCreate which returns context which will have all
    the info set by the user.

    Parameters
    ----------
    `batch_size` : int, optional, default = -1
        Batch size of the pipeline. Negative values for this parameter
        are invalid - the default value may only be used with
        serialized pipeline (the value stored in serialized pipeline
        is used instead).
    `num_threads` : int, optional, default = -1
        Number of CPU threads used by the pipeline.
        Negative values for this parameter are invalid - the default
        value may only be used with serialized pipeline (the value
        stored in serialized pipeline is used instead).
    `device_id` : int, optional, default = -1
        Id of GPU used by the pipeline.
        Negative values for this parameter are invalid - the default
        value may only be used with serialized pipeline (the value
        stored in serialized pipeline is used instead).
    `seed` : int, optional, default = -1
        Seed used for random number generation. Leaving the default value
        for this parameter results in random seed.
    `exec_pipelined` : bool, optional, default = True
        Whether to execute the pipeline in a way that enables
        overlapping CPU and GPU computation, typically resulting
        in faster execution speed, but larger memory consumption.
    `prefetch_queue_depth` : int or {"cpu_size": int, "gpu_size": int}, optional, default = 2
        Depth of the executor pipeline. Deeper pipeline makes ROCAL
        more resistant to uneven execution time of each batch, but it
        also consumes more memory for internal buffers.
        Specifying a dict:
        ``{ "cpu_size": x, "gpu_size": y }``
        instead of an integer will cause the pipeline to use separated
        queues executor, with buffer queue size `x` for cpu stage
        and `y` for mixed and gpu stages. It is not supported when both `exec_async`
        and `exec_pipelined` are set to `False`.
        Executor will buffer cpu and gpu stages separatelly,
        and will fill the buffer queues when the first :meth:`nvidia.dali.pipeline.Pipeline.run`
        is issued.
    `exec_async` : bool, optional, default = True
        Whether to execute the pipeline asynchronously.
        This makes :meth:`nvidia.dali.pipeline.Pipeline.run` method
        run asynchronously with respect to the calling Python thread.
        In order to synchronize with the pipeline one needs to call
        :meth:`nvidia.dali.pipeline.Pipeline.outputs` method.
    `bytes_per_sample` : int, optional, default = 0
        A hint for ROCAL for how much memory to use for its tensors.
    `set_affinity` : bool, optional, default = False
        Whether to set CPU core affinity to the one closest to the
        GPU being used.
    `max_streams` : int, optional, default = -1
        Limit the number of CUDA streams used by the executor.
        Value of -1 does not impose a limit.
        This parameter is currently unused (and behavior of
        unrestricted number of streams is assumed).
    `default_cuda_stream_priority` : int, optional, default = 0
        CUDA stream priority used by ROCAL. See `cudaStreamCreateWithPriority` in CUDA documentation
    """
    '''.
    Args: batch_size
          rocal_cpu
          gpu_id (default 0)
          cpu_threads (default 1)
    This returns a context'''
    _handle = None
    _current_pipeline = None

    def __init__(self, batch_size=-1, num_threads=-1, device_id=-1, seed=-1,
                 exec_pipelined=True, prefetch_queue_depth=2,
                 exec_async=True, bytes_per_sample=0,
                 rocal_cpu=False, max_streams=-1, default_cuda_stream_priority=0, tensor_layout = types.NCHW, reverse_channels = False, multiplier = [1.0,1.0,1.0], offset = [0.0, 0.0, 0.0], tensor_dtype=types.FLOAT):
        if(rocal_cpu):
            # print("comes to cpu")
            self._handle = b.rocalCreate(
                batch_size, types.CPU, device_id, num_threads,prefetch_queue_depth,types.FLOAT)
        else:
            print("comes to gpu")
            self._handle = b.rocalCreate(
                batch_size, types.GPU, device_id, num_threads,prefetch_queue_depth,types.FLOAT)
        if(b.getStatus(self._handle) == types.OK):
            print("Pipeline has been created succesfully")
        else:
            raise Exception("Failed creating the pipeline")
        self._check_ops = ["CropMirrorNormalize"]
        self._check_crop_ops = ["Resize"]
        self._check_ops_decoder = ["ImageDecoder", "ImageDecoderSlice" , "ImageDecoderRandomCrop", "ImageDecoderRaw"]
        self._check_ops_reader = ["FileReader", "TFRecordReaderClassification", "TFRecordReaderDetection",
            "COCOReader", "Caffe2Reader", "Caffe2ReaderDetection", "CaffeReader", "CaffeReaderDetection"]
        self._batch_size = batch_size
        self._num_threads = num_threads
        self._device_id = device_id
        self._seed = seed
        self._exec_pipelined = exec_pipelined
        self._prefetch_queue_depth = prefetch_queue_depth
        self._exec_async = exec_async
        self._bytes_per_sample = bytes_per_sample
        self._rocal_cpu = rocal_cpu
        self._max_streams = max_streams
        self._default_cuda_stream_priority = default_cuda_stream_priority
        self._tensor_layout = tensor_layout
        self._tensor_dtype = tensor_dtype
        self._multiplier = multiplier
        self._reverse_channels = reverse_channels
        self._offset = offset
        self._img_h = None
        self._img_w = None
        self._shuffle = None
        self._name = None
        self._anchors = None
        self._BoxEncoder = None
        self._encode_tensor = None
        self._numOfClasses = None
        self._oneHotEncoding = False
        self._castLabels = False
        self._prev_input_image = None
        self._current_output_image = None
        self._current_pipeline
        self._reader = None
        self._define_graph_set = False


    def store_values(self, operator):
        """
            Check if ops is one of those functionality to determine tensor_layout and tensor_dtype and crop.
            If so preserve it in pipeline to use for dali iterator call.
        """
        if(operator.data in self._check_ops):
            self._tensor_layout = operator._output_layout
            self._tensor_dtype = operator._output_dtype
            self._multiplier = list(map(lambda x: 1/x ,operator._std))
            self._offset = list(map(lambda x,y: -(x/y), operator._mean, operator._std))
            #changing operator std and mean to (1,0) to make sure there is no double normalization
            operator._std = [1.0]
            operator._mean = [0.0]
            if operator._crop_h != 0 and operator._crop_w != 0:
                self._img_w = operator._crop_w
                self._img_h = operator._crop_h
        elif(operator.data in self._check_crop_ops):
            self._img_w = operator._resize_x
            self._img_h = operator._resize_y

    def process_calls(self, output_image):
        last_operator = output_image.prev
        temp = output_image
        while(temp.prev is not None):
            if(temp.data in (self._check_ops + self._check_crop_ops + self._check_ops_reader)):
                self.store_values(temp)
            temp = temp.prev
        file_reader = temp
        file_reader.rocal_c_func_call(self._handle)
        self._shuffle = file_reader._random_shuffle
        self._shard_id = file_reader._shard_id
        self._num_shards = file_reader._num_shards
        self._name = file_reader.data
        temp = temp.next
        operator = temp.next
        while(operator.next.next is not None):
            tensor = operator.next
            if(operator.data in self._check_ops_decoder):
                tensor.data = operator.rocal_c_func_call(
                    self._handle, operator.prev.data, self._img_w, self._img_h, self._shuffle, self._shard_id, self._num_shards, False)
            else:
                tensor.data = operator.rocal_c_func_call(
                    self._handle, operator.prev.data, False)
            operator = operator.next.next
        tensor = last_operator.next
        if(operator.data in self._check_ops_decoder):
            tensor.data = operator.rocal_c_func_call(
                self._handle, operator.prev.data, self._img_w, self._img_h, self._shuffle, self._shard_id, self._num_shards, True)
        else:
            tensor.data = operator.rocal_c_func_call(
                self._handle, operator.prev.data, True)
        return tensor.data

    def build(self):
        """Build the pipeline using rocalVerify call
        """
        status = b.rocalVerify(self._handle)
        if(status != types.OK):
            print("Verify graph failed")
            exit(0)
        return self

    def run(self):
        """ Run the pipeline using rocalRun call
        """
        status = b.rocalRun(self._handle)
        if(status != types.OK):
            print("Rocal Run failed")
        return status

    def define_graph(self):
        """This function is defined by the user to construct the
        graph of operations for their pipeline.
        It returns a list of outputs created by calling ROCAL Operators."""
        print("define_graph NotImplemented")
        raise NotImplementedError


    def get_handle(self):
        return self._handle

    def copyImage(self, array):
        out = np.frombuffer(array, dtype=array.dtype)
        b.rocalCopyToOutput(
            self._handle, np.ascontiguousarray(out, dtype=array.dtype))

    def copyToTensor(self, array,  multiplier, offset, reverse_channels, tensor_format, tensor_dtype):

        b.rocalCopyToOutputTensor(self._handle, ctypes.c_void_p(array.data_ptr()), tensor_format, tensor_dtype,
                                    multiplier[0], multiplier[1], multiplier[2], offset[0], offset[1], offset[2], (1 if reverse_channels else 0))

    def encode(self, bboxes_in, labels_in):
        bboxes_tensor = torch.tensor(bboxes_in).float()
        labels_tensor=  torch.tensor(labels_in).long()
        return self._encode_tensor.prev.rocal_c_func_call(self._handle, bboxes_tensor , labels_tensor )

    def GetOneHotEncodedLabels(self, array):
        return b.getOneHotEncodedLabels(self._handle, array, self._numOfClasses)

    def set_outputs(self, *output_list):
        self._output_list_length = len(output_list)
        b.setOutputImages(self._handle,len(output_list),output_list)

    def __enter__(self):
        Pipeline._current_pipeline = self
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass

    def set_seed(self,seed=0):
        return b.setSeed(seed)

    def create_int_param(self,value):
        return b.CreateIntParameter(value)

    def create_float_param(self,value):
        return b.CreateFloatParameter(value)

    def update_int_param(self,value,param):
        b.UpdateIntParameter(value,param)

    def update_float_param(self,value,param):
        b.UpdateFloatParameter(value,param)

    def get_int_value(self,param):
        return b.GetIntValue(param)

    def get_float_value(self,param):
        return b.GetFloatValue(param)

    def GetImageNameLen(self, array):
        return b.getImageNameLen(self._handle, array)

    def GetImageName(self, array_len):
        return b.getImageName(self._handle,array_len)

    def GetImageId(self, array):
        b.getImageId(self._handle, array)

    def GetBoundingBoxCount(self, array):
        return b.getBoundingBoxCount(self._handle, array)

    def GetBBLabels(self, array):
        return b.getBBLabels(self._handle, array)

    def GetBBCords(self, array):
        return b.getBBCords(self._handle, array)

    def getImageLabels(self, array):
        b.getImageLabels(self._handle, array)

    def copyEncodedBoxesAndLables(self, bbox_array, label_array):
        b.rocalCopyEncodedBoxesAndLables(self._handle, bbox_array, label_array)

    def getEncodedBoxesAndLables(self, batch_size, num_anchors):
        return b.rocalGetEncodedBoxesAndLables(self._handle, batch_size, num_anchors)

    def GetImgSizes(self, array):
        return b.getImgSizes(self._handle, array)

    def GetImageLabels(self, array):
        return b.getImageLabels(self._handle, array)


    def GetBoundingBox(self,array):
        return array

    def GetImageNameLength(self,idx):
        return b.getImageNameLen(self._handle,idx)

    def getOutputWidth(self):
        return b.getOutputWidth(self._handle)

    def getOutputHeight(self):
        return b.getOutputHeight(self._handle)

    def getOutputImageCount(self):
        return b.getOutputImageCount(self._handle)

    def getOutputColorFormat(self):
        return b.getOutputColorFormat(self._handle)

    def getRemainingImages(self):
        return b.getRemainingImages(self._handle)

    def rocalResetLoaders(self):
        return b.rocalResetLoaders(self._handle)

    def isEmpty(self):
        return b.isEmpty(self._handle)

    def Timing_Info(self):
        return b.getTimingInfo(self._handle)
