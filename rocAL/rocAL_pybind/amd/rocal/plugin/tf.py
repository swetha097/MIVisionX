# import numpy as np
# import rocal_pybind as b
# import amd.rocal.types as types
# class ROCALGenericImageIterator(object):
#     def __init__(self, pipeline):
#         self.loader = pipeline
#         self.w = b.getOutputWidth(self.loader._handle)
#         self.h = b.getOutputHeight(self.loader._handle)
#         self.n = b.getOutputImageCount(self.loader._handle)
#         color_format = b.getOutputColorFormat(self.loader._handle)
#         self.p = (1 if (color_format == int(types.GRAY)) else 3)
#         height = self.h*self.n
#         self.out_tensor = None
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
#         return self.out_image , self.out_tensor

#     def reset(self):
#         b.raliResetLoaders(self.loader._handle)

#     def __iter__(self):
#         return self

# class ROCALGenericIteratorDetection(object):
#     def __init__(self, pipeline, tensor_layout = types.NCHW, reverse_channels = False, multiplier = [1.0,1.0,1.0], offset = [0.0, 0.0, 0.0], tensor_dtype=types.FLOAT):
#         self.loader = pipeline
#         self.tensor_format =tensor_layout
#         self.multiplier = multiplier
#         self.offset = offset
#         self.reverse_channels = reverse_channels
#         self.tensor_dtype = tensor_dtype
#         self.w = b.getOutputWidth(self.loader._handle)
#         self.h = b.getOutputHeight(self.loader._handle)
#         self.n = b.getOutputImageCount(self.loader._handle)
#         self.bs = pipeline._batch_size
#         color_format = b.getOutputColorFormat(self.loader._handle)
#         self.p = (1 if (color_format == int(types.GRAY)) else 3)

#         if self.tensor_dtype == types.FLOAT:
#             self.out = np.zeros(( self.bs*self.n, self.p, int(self.h/self.bs), self.w,), dtype = "float32")
#         elif self.tensor_dtype == types.FLOAT16:
#             self.out = np.zeros(( self.bs*self.n, self.p, int(self.h/self.bs), self.w,), dtype = "float16")
#         # self.labels = np.zeros((self.bs),dtype = "int32")

#     def next(self):
#         return self.__next__()

#     def __next__(self):
#         if(b.isEmpty(self.loader._handle)):
#             timing_info = b.getTimingInfo(self.loader._handle)
#             print("Load     time ::",timing_info.load_time)
#             print("Decode   time ::",timing_info.decode_time)
#             print("Process  time ::",timing_info.process_time)
#             print("Transfer time ::",timing_info.transfer_time)
#             raise StopIteration

#         if self.loader.run() != 0:
#             raise StopIteration

#         if(types.NCHW == self.tensor_format):
#             self.loader.copyToTensorNCHW(self.out, self.multiplier, self.offset, self.reverse_channels, int(self.tensor_dtype))
#         else:
#             self.loader.copyToTensorNHWC(self.out, self.multiplier, self.offset, self.reverse_channels, int(self.tensor_dtype))
        
#         if(self.loader._name == "TFRecordReaderDetection"):
#             self.bbox_list =[]
#             self.label_list=[]
#             self.num_bboxes_list=[]
#             #Count of labels/ bboxes in a batch
#             self.bboxes_label_count = np.zeros(self.bs, dtype="int32")
#             self.count_batch = self.loader.GetBoundingBoxCount(self.bboxes_label_count)
#             self.num_bboxes_list = self.bboxes_label_count.tolist()
#             # 1D labels array in a batch
#             self.labels = np.zeros(self.count_batch, dtype="int32")
#             self.loader.GetBBLabels(self.labels)
#             # 1D bboxes array in a batch
#             self.bboxes = np.zeros((self.count_batch*4), dtype="float32")
#             self.loader.GetBBCords(self.bboxes)
#             #1D Image sizes array of image in a batch
#             self.img_size = np.zeros((self.bs * 2),dtype = "int32")
#             self.loader.GetImgSizes(self.img_size)
#             count =0 # number of bboxes per image
#             sum_count=0 # sum of the no. of the bboxes
#             for i in range(self.bs):
#                 count = self.bboxes_label_count[i]
#                 self.label_2d_numpy = (self.labels[sum_count : sum_count+count])
#                 self.label_2d_numpy = np.reshape(self.label_2d_numpy, (-1, 1)).tolist()
#                 self.bb_2d_numpy = (self.bboxes[sum_count*4 : (sum_count+count)*4])
#                 self.bb_2d_numpy = np.reshape(self.bb_2d_numpy, (-1, 4)).tolist()
#                 self.label_list.append(self.label_2d_numpy)
#                 self.bbox_list.append(self.bb_2d_numpy)
#                 sum_count = sum_count +count

#             self.target = self.bbox_list
#             self.target1 = self.label_list
#             max_cols = max([len(row) for batch in self.target for row in batch])
#             # max_rows = max([len(batch) for batch in self.target])
#             max_rows = 100
#             bb_padded = [batch + [[0] * (max_cols)] * (max_rows - len(batch)) for batch in self.target]
#             bb_padded_1=[row + [0] * (max_cols - len(row)) for batch in bb_padded for row in batch]
#             arr = np.asarray(bb_padded_1)
#             self.res = np.reshape(arr, (-1, max_rows, max_cols))
#             max_cols = max([len(row) for batch in self.target1 for row in batch])
#             # max_rows = max([len(batch) for batch in self.target1])
#             max_rows = 100
#             lab_padded = [batch + [[0] * (max_cols)] * (max_rows - len(batch)) for batch in self.target1]
#             lab_padded_1=[row + [0] * (max_cols - len(row)) for batch in lab_padded for row in batch]
#             labarr = np.asarray(lab_padded_1)
#             self.l = np.reshape(labarr, (-1, max_rows, max_cols))
#             self.num_bboxes_arr = np.array(self.num_bboxes_list)

#             if self.tensor_dtype == types.FLOAT:
#                 return self.out.astype(np.float32), self.res, self.l, self.num_bboxes_arr
#             elif self.tensor_dtype == types.FLOAT16:
#                 return self.out.astype(np.float16), self.res, self.l, self.num_bboxes_arr
#         elif (self.loader._name == "TFRecordReaderClassification"):
#             if(self.loader._oneHotEncoding == True):
#                 self.labels = np.zeros((self.bs)*(self.loader._numOfClasses),dtype = "int32")
#                 self.loader.GetOneHotEncodedLabels(self.labels)
#                 self.labels = np.reshape(self.labels, (-1, self.bs, self.loader._numOfClasses))
#             else:
#                 self.labels = np.zeros((self.bs),dtype = "int32")
#                 self.loader.GetImageLabels(self.labels)

#             if self.tensor_dtype == types.FLOAT:
#                 return self.out.astype(np.float32), self.labels
#             elif self.tensor_dtype == types.TensorDataType.FLOAT16:
#                 return self.out.astype(np.float16), self.labels
        
#     def reset(self):
#         b.raliResetLoaders(self.loader._handle)

#     def __iter__(self):
#         return self


# class ROCALIterator(ROCALGenericIteratorDetection):
#     """
#     ROCAL iterator for detection and classification tasks for PyTorch. It returns 2 or 3 outputs
#     (data and label) or (data , bbox , labels) in the form of PyTorch's Tensor.
#     Calling
#     .. code-block:: python
#        ROCALIterator(pipelines, size)
#     is equivalent to calling
#     .. code-block:: python
#        ROCALGenericIteratorDetection(pipelines, ["data", "label"], size)


#     """
#     def __init__(self,
#                  pipelines,
#                  size = 0,
#                  auto_reset=False,
#                  fill_last_batch=True,
#                  dynamic_shape=False,
#                  last_batch_padded=False):
#         pipe = pipelines
#         super(ROCALIterator, self).__init__(pipe, tensor_layout = pipe._tensor_layout, tensor_dtype = pipe._tensor_dtype,
#                                                             multiplier=pipe._multiplier, offset=pipe._offset)



# class ROCAL_iterator(ROCALGenericImageIterator):
#     """
#     ROCAL iterator for classification tasks for PyTorch. It returns 2 outputs
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
#         super(ROCAL_iterator, self).__init__(pipe)


import numpy as np
import rocal_pybind as b
import amd.rocal.types as types
class ROCALGenericImageIterator(object):
    def __init__(self, pipeline):
        self.loader = pipeline
        self.w = b.getOutputWidth(self.loader._handle)
        self.h = b.getOutputHeight(self.loader._handle)
        self.n = b.getOutputImageCount(self.loader._handle)
        color_format = b.getOutputColorFormat(self.loader._handle)
        self.p = (1 if (color_format == int(types.GRAY)) else 3)
        height = self.h*self.n
        self.out_tensor = None
        self.out_image = np.zeros((height, self.w, self.p), dtype = "uint8")
        self.bs = pipeline._batch_size

    def next(self):
        return self.__next__()

    def __next__(self):
        if b.getRemainingImages(self.loader._handle) < self.bs:
            raise StopIteration

        if self.loader.run() != 0:
            raise StopIteration

        self.loader.copyImage(self.out_image)
        return self.out_image , self.out_tensor

    def reset(self):
        b.rocalResetLoaders(self.loader._handle)

    def __iter__(self):
        return self

class ROCALGenericIteratorDetection(object):
    def __init__(self, pipeline, tensor_layout = types.NCHW, reverse_channels = False, multiplier = [1.0,1.0,1.0], offset = [0.0, 0.0, 0.0], tensor_dtype=types.FLOAT):
        self.loader = pipeline
        self.tensor_format =tensor_layout
        self.multiplier = multiplier
        self.offset = offset
        self.reverse_channels = reverse_channels
        self.tensor_dtype = tensor_dtype
        print("INIT THE ITERATOR!!!!")
        self.len = b.getRemainingImages(self.loader._handle)
        print(self.len)
        if self.loader._name is None:
            self.loader._name = self.loader._reader
        

    def next(self):
        return self.__next__()

    def __next__(self):
        if(b.isEmpty(self.loader._handle)):
            timing_info = b.getTimingInfo(self.loader._handle)
            print("Load     time ::",timing_info.load_time)
            print("Decode   time ::",timing_info.decode_time)
            print("Process  time ::",timing_info.process_time)
            print("Transfer time ::",timing_info.transfer_time)
            raise StopIteration
        print("IN NEXT FUNCTION !!!")
        if self.loader.rocalRun() != 0:
            raise StopIteration
        else:
            self.output_tensor_list = self.loader.rocalGetOutputTensors()

        print(self.output_tensor_list)
        self.augmentation_count = len(self.output_tensor_list)
        print("AUG COUNT", self.augmentation_count)
        self.w = self.output_tensor_list[0].batch_width() #2000
        self.h = self.output_tensor_list[0].batch_height()  #2000
        self.batch_size = self.output_tensor_list[0].batch_size()  # 1
        self.color_format = self.output_tensor_list[0].color_format()  # 3
        print(self.color_format)
        print(self.batch_size , self.h , self.w, self.color_format )
        self.out = np.zeros(( self.batch_size * self.augmentation_count, self.h, self.w, self.color_format),dtype="uint8")
        # self.output = torch.empty((self.batch_size, self.h, self.w, self.color_format,), dtype=torch.uint8)
        # self.out = torch.permute(self.output, (0,3,1,2))
        if(self.loader._name == "TFRecordReaderDetection"):
            self.bbox_list =[]
            self.label_list=[]
            self.num_bboxes_list=[]
            
            # std::cerr<<"self.output_tensor_list[0] "<<self.output_tensor_list[0];
            # print("self.output_tensor_list[0]  ",self.output_tensor_list[0])
            self.output_tensor_list[0].copy_data_numpy(self.out)
            print("self.out", self.out)
            # print("after copy_data_numpy")
            #Count of labels/ bboxes in a batch
            self.labels=self.loader.rocalGetBoundingBoxLabel()
            # print("labels    ", self.labels)
            # print("labels    ", len(self.labels[0]))
            # print("labels    ", self.labels[1].shape)
            # print("labels    ", self.labels[2].shape)
            
            self.bboxes =self.loader.rocalGetBoundingBoxCords()
            # print("bbox_list    ", self.bboxes)
            self.img_size = np.zeros((self.batch_size * 2),dtype = "int32")
            self.loader.GetImgSizes(self.img_size)
            # print("self.img_size",self.img_size)
            # self.bboxes_label_count = np.zeros(self.bs, dtype="int32")
            # self.count_batch = self.loader.GetBoundingBoxCount(self.bboxes_label_count)
            # self.num_bboxes_list = self.bboxes_label_count.tolist()
            # # 1D labels array in a batch
            # self.labels = np.zeros(self.count_batch, dtype="int32")
            # self.loader.GetBBLabels(self.labels)
            # # 1D bboxes array in a batch
            # self.bboxes = np.zeros((self.count_batch*4), dtype="float32")
            # self.loader.GetBBCords(self.bboxes)
            # #1D Image sizes array of image in a batch
            # self.img_size = np.zeros((self.bs * 2),dtype = "int32")
            # self.loader.GetImgSizes(self.img_size)
            count =0 # number of bboxes per image
            lab_list=[]
            sum_count=0 # sum of the no. of the bboxes
            # print("self.batch_size",self.batch_size)
            for i in range(self.batch_size):
                count = len(self.labels[i])
                lab_list.append(count)
                self.label_2d_numpy = self.labels[i]
                
                # print("self.label_2d_numpy",self.label_2d_numpy)                
                self.label_2d_numpy = np.reshape(self.label_2d_numpy, (-1, 1)).tolist()
                # print("self.label_2d_numpy",self.label_2d_numpy)                 
                self.bb_2d_numpy = (self.bboxes[i])
                # print("self.bb_2d_numpy",self.bb_2d_numpy)
                self.bb_2d_numpy = np.reshape(self.bb_2d_numpy, (-1, 4)).tolist()
                # print("self.bb_2d_numpy",self.bb_2d_numpy)
                
                self.label_list.append(self.label_2d_numpy)
                self.bbox_list.append(self.bb_2d_numpy)
                sum_count = sum_count +count
                # print("&&&",count)
            # print("sum_count",lab_list)
            self.target = self.bbox_list
            self.target1 = self.label_list
            max_cols = max([len(row) for batch in self.target for row in batch])
            # max_rows = max([len(batch) for batch in self.target])
            max_rows = 100
            bb_padded = [batch + [[0] * (max_cols)] * (max_rows - len(batch)) for batch in self.target]
            bb_padded_1=[row + [0] * (max_cols - len(row)) for batch in bb_padded for row in batch]
            arr = np.asarray(bb_padded_1)
            self.res = np.reshape(arr, (-1, max_rows, max_cols))
            max_cols = max([len(row) for batch in self.target1 for row in batch])
            # max_rows = max([len(batch) for batch in self.target1])
            max_rows = 100
            lab_padded = [batch + [[0] * (max_cols)] * (max_rows - len(batch)) for batch in self.target1]
            lab_padded_1=[row + [0] * (max_cols - len(row)) for batch in lab_padded for row in batch]
            labarr = np.asarray(lab_padded_1)
            label_list= np.array(lab_list)
            self.l = np.reshape(labarr, (-1, max_rows, max_cols))
            # self.num_bboxes_arr = np.array(self.num_bboxes_list)
            # print("self.res",self.res)
            # print("self.l",self.l)
            # print("label_list",label_list)
        
            print("SELF.OUT",self.out)
            if self.tensor_dtype == types.FLOAT:
                return self.out.astype(np.float32), self.res, self.l,label_list
            elif self.tensor_dtype == types.FLOAT16:
                return self.out.astype(np.float16), self.res, self.l,label_list
        elif (self.loader._name == "TFRecordReaderClassification"):
            print("CLASSIFICATION ITERATOR")
            print(self.output_tensor_list)
            self.output_tensor_list[0].copy_data_numpy(self.out)
            self.labels = self.loader.rocalGetImageLabels()#numpy
            # self.labels_tensor = torch.from_numpy(self.labels).type(torch.LongTensor)
            return (self.out),self.labels
    def reset(self):
        b.rocalResetLoaders(self.loader._handle)

    def __iter__(self):
        return self


class ROCALIterator(ROCALGenericIteratorDetection):
    """
    ROCAL iterator for detection and classification tasks for PyTorch. It returns 2 or 3 outputs
    (data and label) or (data , bbox , labels) in the form of PyTorch's Tensor.
    Calling
    .. code-block:: python
       ROCALIterator(pipelines, size)
    is equivalent to calling
    .. code-block:: python
       ROCALGenericIteratorDetection(pipelines, ["data", "label"], size)


    """
    def __init__(self,
                 pipelines,
                 size = 0,
                 auto_reset=False,
                 fill_last_batch=True,
                 dynamic_shape=False,
                 last_batch_padded=False):
        pipe = pipelines
        super(ROCALIterator, self).__init__(pipe, tensor_layout = pipe._tensor_layout, tensor_dtype = pipe._tensor_dtype,
                                                            multiplier=pipe._multiplier, offset=pipe._offset)



class ROCAL_iterator(ROCALGenericImageIterator):
    """
    ROCAL iterator for classification tasks for PyTorch. It returns 2 outputs
    (data and label) in the form of PyTorch's Tensor.
   
    """
    def __init__(self,
                 pipelines,
                 size = 0,
                 auto_reset=False,
                 fill_last_batch=True,
                 dynamic_shape=False,
                 last_batch_padded=False):
        pipe = pipelines
        super(ROCAL_iterator, self).__init__(pipe)