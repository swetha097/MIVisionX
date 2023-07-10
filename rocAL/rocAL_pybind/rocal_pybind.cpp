/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include "api/rocal_api_types.h"
#include "rocal_api.h"
#include "tensor.h"
#include "api/rocal_api_parameters.h"
#include "api/rocal_api_data_loaders.h"
#include "api/rocal_api_augmentation.h"
#include "api/rocal_api_data_transfer.h"
#include "api/rocal_api_info.h"
namespace py = pybind11;

using float16 = half_float::half;
static_assert(sizeof(float16) == 2, "Bad size");
namespace pybind11
{
    namespace detail
    {
        constexpr int NPY_FLOAT16 = 23;
        // Kinda following: https://github.com/pybind/pybind11/blob/9bb3313162c0b856125e481ceece9d8faa567716/include/pybind11/numpy.h#L1000
        template <>
        struct npy_format_descriptor<float16>
        {
            static constexpr auto name = _("float16");
            static pybind11::dtype dtype()
            {
                handle ptr = npy_api::get().PyArray_DescrFromType_(NPY_FLOAT16);
                return reinterpret_borrow<pybind11::dtype>(ptr);
            }
        };
    }
} // namespace pybind11::detail
namespace rocal
{
    using namespace pybind11::literals; // NOLINT
    // PYBIND11_MODULE(rocal_backend_impl, m) {
    static void *ctypes_void_ptr(const py::object &object)
    {
        auto ptr_as_int = getattr(object, "value", py::none());
        if (ptr_as_int.is_none())
        {
            return nullptr;
        }
        void *ptr = PyLong_AsVoidPtr(ptr_as_int.ptr());
        return ptr;
    }

    py::object wrapper_image_name_length(RocalContext context, py::array_t<int> array)
    {
        auto buf = array.request();
        int *ptr = (int *)buf.ptr;
        // call pure C++ function
        int length = rocalGetImageNameLen(context, ptr);
        return py::cast(length);
    }

    py::object wrapper_image_name(RocalContext context, int array_len)
    {
        py::array_t<char> array;
        auto buf = array.request();
        char *ptr = (char *)buf.ptr;
        ptr = (char *)calloc(array_len, sizeof(char));
        // call pure C++ function
        rocalGetImageName(context, ptr);
        std::string s(ptr);
        free(ptr);
        return py::bytes(s);
    }

    py::object wrapper_tensor(RocalContext context, py::object p,
                                RocalTensorLayout tensor_format, RocalTensorOutputType tensor_output_type, float multiplier0,
                                float multiplier1, float multiplier2, float offset0,
                                float offset1, float offset2,
                                bool reverse_channels)
    {
        auto ptr = ctypes_void_ptr(p);
        // call pure C++ function

        int status = rocalCopyToOutputTensor(context, ptr, tensor_format, tensor_output_type, multiplier0,
                                              multiplier1, multiplier2, offset0,
                                              offset1, offset2, reverse_channels);
        // std::cerr<<"\n Copy failed with status :: "<<status;
        return py::cast<py::none>(Py_None);
    }

    PYBIND11_MODULE(rocal_pybind, m)
    {
        m.doc() = "Python bindings for the C++ portions of ROCAL";
        // rocal_api.h
        m.def("rocalCreate", &rocalCreate, "Creates context with the arguments sent and returns it",
              py::return_value_policy::reference);
            //   py::arg("batch_size"),
            //   py::arg("affinity"),
            //   py::arg("gpu_id") = 0,
            //   py::arg("cpu_thread_count") = 1,
            //   py::arg("prefetch_queue_depth") = 3,
            //   py::arg("output_data_type") = 0);
        m.def("rocalVerify", &rocalVerify);
        m.def("rocalRun", &rocalRun);
        m.def("rocalRelease", &rocalRelease);
        // rocal_api_types.h
        py::class_<TimingInfo>(m, "TimingInfo")
            .def_readwrite("load_time", &TimingInfo::load_time)
            .def_readwrite("decode_time", &TimingInfo::decode_time)
            .def_readwrite("process_time", &TimingInfo::process_time)
            .def_readwrite("transfer_time", &TimingInfo::transfer_time);
        py::class_<rocalTensor>(m, "rocalTensor")
                .def(
                "__add__",
                [](rocalTensor *output_tensor, rocalTensor *output_tensor1)
                {
                    py::object fn_module = py::module::import("amd.rocal.fn");
                    auto fn_function_call = fn_module.attr("tensor_add_tensor_float")(output_tensor, output_tensor1).cast<RocalTensor>();
                    return fn_function_call;
                },
                R"code(
                Adds a node for arithmetic operation
                )code", py::return_value_policy::reference
            )
                .def(
                "__mul__",
                [](rocalTensor *output_tensor, float scalar)
                {
                    py::object fn_module = py::module::import("amd.rocal.fn");
                    auto fn_function_call = fn_module.attr("tensor_mul_scalar_float")(output_tensor, "scalar"_a=scalar).cast<RocalTensor>();
                    return fn_function_call;
                },
                R"code(
                Returns a tensor ROI
                Adds a node for arithmetic operation
                )code", py::return_value_policy::reference
            )
                .def(
                "get_roi_at",
                [](rocalTensor &output_tensor, uint idx)
                {
                    return std::make_pair(output_tensor.info().get_roi()[idx].x1, output_tensor.info().get_roi()[idx].y1);
                },
                R"code(
                Returns a tensor ROI
                ex : width, height in case of an image data
                ex : samples , channels in case of an audio data
                )code"
            )
                .def(
                "get_rois",
                [](rocalTensor &output_tensor)
                {
                    return py::array(py::buffer_info(
                            (int *)(output_tensor.info().get_roi()),
                            sizeof(int),
                            py::format_descriptor< int>::format(),
                            1,
                            {output_tensor.info().dims().at(0) * 4},
                            {sizeof(int) }));
                },
                R"code(
                Returns a tensor ROI
                ex : width, height in case of an image data
                ex : samples , channels in case of an audio data
                )code"
            )
                .def(
                "num_of_dims",
                [](rocalTensor &output_tensor)
                {
                    return output_tensor.info().num_of_dims();
                },
                R"code(
                Returns a tensor data's total number of dimensions.
                ex: 3 in case of audio, 4 in case of an image, 5 in case of video 
                )code"
            )
                .def(
                "batch_height",
                [](rocalTensor &output_tensor)
                {
                    return output_tensor.info().max_shape().at(1);
                },
                R"code(
                Returns a tensor buffer's height.
                )code"
            )
            .def(
                "batch_width",
                [](rocalTensor &output_tensor)
                {
                    return output_tensor.info().max_shape().at(0);
                },
                R"code(
                Returns a tensor buffer's width.
                )code"
            )
            .def(
                "batch_size",
                [](rocalTensor &output_tensor)
                {
                    return output_tensor.info().dims().at(0);
                },
                R"code(
                Returns a tensor batch size.
                )code"
            )
            .def(
                "color_format",
                [](rocalTensor &output_tensor)
                {
                    if ((output_tensor.info().color_format() == RocalColorFormat::RGB24) || (output_tensor.info().color_format() == RocalColorFormat::BGR24))
                        return 3;
                    else
                        return 1;
                },
                R"code(
                Returns a tensor batch size.
                )code"
            )
            .def(
            "copy_data", [](rocalTensor &output_tensor, py::object p)
            {
            auto ptr = ctypes_void_ptr(p);
            output_tensor.copy_data(ptr);
            }
            )
            .def(
            "copy_data", [](rocalTensor &output_tensor, py::object p, uint last_batch_size)
            {
            auto ptr = ctypes_void_ptr(p);
            output_tensor.copy_data(ptr, last_batch_size);
            }
            )
            .def(
            "copy_data", [](rocalTensor &output_tensor, py::object p, uint max_x1, uint max_y1)
            {
            auto ptr = ctypes_void_ptr(p);
            output_tensor.copy_data(ptr, max_x1, max_y1);
            }
            ,py::return_value_policy::reference
            )
            .def(
                "at",
                [](rocalTensor &output_tensor, uint idx)
                {
                    uint h = output_tensor.info().max_shape().at(1);
                    uint w = output_tensor.info().max_shape().at(0);

                    if (output_tensor.info().layout() == RocalTensorlayout::NHWC)
                    {
                        unsigned c = output_tensor.info().dims().at(3);
                        return py::array(py::buffer_info(
                            ((unsigned char *)(output_tensor.buffer())) + idx * c * h * w,
                            sizeof(unsigned char),
                            py::format_descriptor<unsigned char>::format(),
                            output_tensor.info().num_of_dims() - 1,
                            {h, w, c},
                            {sizeof(unsigned char) * w * c, sizeof(unsigned char) * c, sizeof(unsigned char)}));
                    }

                    else if (output_tensor.info().layout() == RocalTensorlayout::NCHW)
                    {
                        unsigned n = output_tensor.info().dims().at(0);
                        unsigned c = output_tensor.info().dims().at(1);
                        return py::array(py::buffer_info(
                            ((unsigned char *)(output_tensor.buffer())) + idx * c * h * w,
                            sizeof(unsigned char),
                            py::format_descriptor<unsigned char>::format(),
                            output_tensor.info().num_of_dims(),
                            {c, h, w},
                            {sizeof(unsigned char) * c * h * w, sizeof(unsigned char) * h * w, sizeof(unsigned char) * w, sizeof(unsigned char)}));
                    }
                },
                "idx"_a,
                R"code(
                Returns a rocAL tensor at given position `i` in the rocalTensorlist.
                )code",
                py::keep_alive<0, 1>());
        py::class_<rocalTensorList>(m, "rocalTensorList")
            .def(
                "__getitem__",
                [](rocalTensorList &output_tensor_list, uint idx)
                {
                    return output_tensor_list.at(idx);
                },
                R"code(
                Returns a tensor at given position in the list.
                )code")

            .def("at",
                [](rocalTensorList &output_tensor_list, uint idx)
                {
                    uint h = output_tensor_list.at(idx)->info().max_shape().at(1);
                    uint w = output_tensor_list.at(idx)->info().max_shape().at(0);

                    if (output_tensor_list.at(idx)->info().layout() == RocalTensorlayout::NHWC)
                    {
                        unsigned n = output_tensor_list.at(idx)->info().dims().at(0);
                        unsigned c = output_tensor_list.at(idx)->info().dims().at(3);
                        return py::array(py::buffer_info(
                            (unsigned char *)(output_tensor_list.at(idx)->buffer()),
                            sizeof(unsigned char),
                            py::format_descriptor<unsigned char>::format(),
                            output_tensor_list.at(idx)->info().num_of_dims(),
                            {n, h, w, c},
                            {sizeof(unsigned char) * w * h * c, sizeof(unsigned char) * w * c, sizeof(unsigned char) * c, sizeof(unsigned char)}));
                    }

                    else if (output_tensor_list.at(idx)->info().layout() == RocalTensorlayout::NCHW)
                    {
                        unsigned n = output_tensor_list.at(idx)->info().dims().at(0);
                        unsigned c = output_tensor_list.at(idx)->info().dims().at(1);
                        return py::array(py::buffer_info(
                            (unsigned char *)(output_tensor_list.at(idx)->buffer()),
                            sizeof(unsigned char),
                            py::format_descriptor<unsigned char>::format(),
                            output_tensor_list.at(idx)->info().num_of_dims(),
                            {n, c, h, w},
                            {sizeof(unsigned char) * c * h * w, sizeof(unsigned char) * h * w, sizeof(unsigned char) * w, sizeof(unsigned char)}));
                    }
                },
                "idx"_a,
                R"code(
                Returns a rocAL tensor at given position `i` in the rocalTensorlist.
                )code",
                py::keep_alive<0, 1>());
        py::class_<rocalTensorInfo>(m, "rocalTensorInfo");

        py::module types_m = m.def_submodule("types");
        types_m.doc() = "Datatypes and options used by ROCAL";
        py::enum_<RocalStatus>(types_m, "RocalStatus", "Status info")
            .value("OK", ROCAL_OK)
            .value("CONTEXT_INVALID", ROCAL_CONTEXT_INVALID)
            .value("RUNTIME_ERROR", ROCAL_RUNTIME_ERROR)
            .value("UPDATE_PARAMETER_FAILED", ROCAL_UPDATE_PARAMETER_FAILED)
            .value("INVALID_PARAMETER_TYPE", ROCAL_INVALID_PARAMETER_TYPE)
            .export_values();
        py::enum_<RocalProcessMode>(types_m, "RocalProcessMode", "Processing mode")
            .value("GPU", ROCAL_PROCESS_GPU)
            .value("CPU", ROCAL_PROCESS_CPU)
            .export_values();
        py::enum_<RocalTensorOutputType>(types_m, "RocalTensorOutputType", "Tensor types")
            .value("FLOAT", ROCAL_FP32)
            .value("FLOAT16", ROCAL_FP16)
            .value("UINT8", ROCAL_UINT8)
            .export_values();
        py::enum_<RocalImageSizeEvaluationPolicy>(types_m, "RocalImageSizeEvaluationPolicy", "Decode size policies")
            .value("MAX_SIZE", ROCAL_USE_MAX_SIZE)
            .value("USER_GIVEN_SIZE", ROCAL_USE_USER_GIVEN_SIZE)
            .value("MOST_FREQUENT_SIZE", ROCAL_USE_MOST_FREQUENT_SIZE)
            .value("MAX_SIZE_ORIG", ROCAL_USE_MAX_SIZE_RESTRICTED)
            .value("USER_GIVEN_SIZE_ORIG", ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED)
            .export_values();
        py::enum_<RocalImageColor>(types_m, "RocalImageColor", "Image type")
            .value("RGB", ROCAL_COLOR_RGB24)
            .value("BGR", ROCAL_COLOR_BGR24)
            .value("GRAY", ROCAL_COLOR_U8)
            .value("RGB_PLANAR", ROCAL_COLOR_RGB_PLANAR)
            .export_values();
        py::enum_<RocalTensorLayout>(types_m, "RocalTensorLayout", "Tensor layout type")
            .value("NHWC", ROCAL_NHWC)
            .value("NCHW", ROCAL_NCHW)
            .export_values();
        py::enum_<RocalDecodeDevice>(types_m, "RocalDecodeDevice", "Decode device type")
            .value("HARDWARE_DECODE", ROCAL_HW_DECODE)
            .value("SOFTWARE_DECODE", ROCAL_SW_DECODE)
            .export_values();
        py::enum_<RocalDecoderType>(types_m,"RocalDecoderType", "Rocal Decoder Type")
            .value("DECODER_TJPEG",ROCAL_DECODER_TJPEG)
            .value("DECODER_OPENCV",ROCAL_DECODER_OPENCV)
            .value("DECODER_HW_JEPG",ROCAL_DECODER_HW_JPEG)
            .value("DECODER_VIDEO_FFMPEG_SW",ROCAL_DECODER_VIDEO_FFMPEG_SW)
            .value("DECODER_VIDEO_FFMPEG_HW",ROCAL_DECODER_VIDEO_FFMPEG_HW)
            .export_values();
        py::enum_<RocalResizeScalingMode>(types_m,"RocalResizeScalingMode","Decode size policies")
            .value("SCALING_MODE_DEFAULT",ROCAL_SCALING_MODE_DEFAULT)
            .value("SCALING_MODE_STRETCH",ROCAL_SCALING_MODE_STRETCH)
            .value("SCALING_MODE_NOT_SMALLER",ROCAL_SCALING_MODE_NOT_SMALLER)
            .value("SCALING_MODE_NOT_LARGER",ROCAL_SCALING_MODE_NOT_LARGER)
            .export_values();
        py::enum_<RocalResizeInterpolationType>(types_m,"RocalResizeInterpolationType","Decode size policies")
            .value("NEAREST_NEIGHBOR_INTERPOLATION",ROCAL_NEAREST_NEIGHBOR_INTERPOLATION)
            .value("LINEAR_INTERPOLATION",ROCAL_LINEAR_INTERPOLATION)
            .value("CUBIC_INTERPOLATION",ROCAL_CUBIC_INTERPOLATION)
            .value("LANCZOS_INTERPOLATION",ROCAL_LANCZOS_INTERPOLATION)
            .value("GAUSSIAN_INTERPOLATION",ROCAL_GAUSSIAN_INTERPOLATION)
            .value("TRIANGULAR_INTERPOLATION",ROCAL_TRIANGULAR_INTERPOLATION)
            .export_values();
         py::enum_<RocalAudioBorderType>(types_m,"RocalAudioBorderType", "Rocal Audio Border Type")
            .value("ZERO",ZERO)
            .value("CLAMP",CLAMP)
            .value("REFLECT",REFLECT)
            .export_values();
        py::enum_<RocalSpectrogramLayout>(types_m,"RocalSpectrogramLayout", "Rocal Audio Border Type")
            .value("FT",FT)
            .value("TF",TF)
            .export_values();
        py::enum_<RocalMelScaleFormula>(types_m,"RocalMelScaleFormula", "Rocal Audio Border Type")
            .value("SLANEY",SLANEY)
            .value("HTK",HTK)
            .export_values();
        py::enum_<RocalOutOfBoundsPolicy>(types_m,"RocalOutOfBoundsPolicy", "Rocal Audio Border Type")
            .value("PAD",PAD)
            .value("TRIMTOSHAPE",TRIMTOSHAPE)
            .value("ERROR",ERROR)
            .export_values();
        py::enum_<RocalLastBatchPolicy>(types_m,"RocalLastBatchPolicy", "Rocal Last Batch Policy")
            .value("LAST_BATCH_FILL",ROCAL_LAST_BATCH_FILL)
            .value("LAST_BATCH_DROP",ROCAL_LAST_BATCH_DROP)
            .value("LAST_BATCH_PARTIAL",ROCAL_LAST_BATCH_PARTIAL)
            .export_values();
        // rocal_api_info.h
        m.def("getRemainingImages", &rocalGetRemainingImages, py::return_value_policy::reference);
        m.def("getLastBatchPaddedSize", &rocalGetLastBatchPaddedSize, py::return_value_policy::reference);
        m.def("isEmpty", &rocalIsEmpty, py::return_value_policy::reference);
        m.def("getStatus", rocalGetStatus, py::return_value_policy::reference);
        m.def("rocalGetErrorMessage", &rocalGetErrorMessage, py::return_value_policy::reference);
        m.def("rocalGetTimingInfo", &rocalGetTimingInfo, py::return_value_policy::reference);
        m.def("setOutputImages", &rocalSetOutputs, py::return_value_policy::reference);
        m.def("getTimingInfo", &rocalGetTimingInfo, py::return_value_policy::reference);
        m.def("labelReader", &rocalCreateLabelReader, py::return_value_policy::reference);
        m.def("labelReaderFileList", &rocalCreateFileListLabelReader, py::return_value_policy::reference);
        m.def("COCOReader", &rocalCreateCOCOReader, py::return_value_policy::reference);
        // rocal_api_meta_data.h
        m.def("RandomBBoxCrop", &rocalRandomBBoxCrop);
        m.def("BoxEncoder",&rocalBoxEncoder);
        m.def("BoxIOUMatcher", &rocalBoxIOUMatcher);
        m.def("getImageId", [](RocalContext context, py::array_t<int> array)
        {
            auto buf = array.request();
            int* ptr = (int*) buf.ptr;
            return rocalGetImageId(context,ptr);
        }
        );
        m.def("getImgSizes", [](RocalContext context, py::array_t<int> array)
        {
            auto buf = array.request();
            int* ptr = (int*) buf.ptr;
            rocalGetImageSizes(context,ptr);
        }
        );
        // rocal_api_parameter.h
        m.def("setSeed", &rocalSetSeed);
        m.def("getSeed", &rocalGetSeed);
        m.def("CreateIntUniformRand", &rocalCreateIntUniformRand);
        m.def("CreateFloatUniformRand", &rocalCreateFloatUniformRand);
        m.def("CreateIntRand", [](std::vector<int> values, std::vector<double> frequencies)
              { return rocalCreateIntRand(values.data(), frequencies.data(), values.size()); });
        m.def("CreateFloatRand", [](std::vector<float> values, std::vector<double> frequencies)
              { return rocalCreateFloatRand(values.data(), frequencies.data(), values.size()); });
        m.def("CreateIntParameter", &rocalCreateIntParameter);
        m.def("CreateFloatParameter", &rocalCreateFloatParameter);
        m.def("UpdateIntParameter", &rocalUpdateIntParameter);
        m.def("UpdateFloatParameter", &rocalUpdateFloatParameter);
        m.def("GetIntValue", &rocalGetIntValue);
        m.def("GetFloatValue", &rocalGetFloatValue);
        m.def("rocalGetBoundingBoxCount", &rocalGetBoundingBoxCount);
        // rocal_api_data_transfer.h
        m.def("rocalCopyToOutputTensor",&wrapper_tensor);
        m.def("rocalGetOutputTensors", [](RocalContext context)
              {
            rocalTensorList * tl = rocalGetOutputTensors(context);
            py::list list;
            unsigned int size_of_tensor_list = tl->size();
            for (uint i =0; i< size_of_tensor_list; i++)
                list.append(tl->at(i));
            return list; });
        m.def(
            "rocalGetImageLabels", [](RocalContext context)
    {
            rocalTensorList *labels = rocalGetImageLabels(context);
            return py::array(py::buffer_info(
                            (int *)(labels->at(0)->buffer()),
                            sizeof(int),
                            py::format_descriptor<int>::format(),
                            1,
                            {labels->size()},
                            {sizeof(int) }));
    }
            );
        m.def(
            "rocalGetBoundingBoxLabel", [](RocalContext context)
    {
            rocalTensorList *labels = rocalGetBoundingBoxLabel(context);
            py::list labels_list;
            py::array_t<int> labels_array;
            for (int i = 0; i < labels->size(); i++) {
                int *labels_buffer = (int *)(labels->at(i)->buffer());
                labels_array = py::array(py::buffer_info(
                            (int *)(labels->at(i)->buffer()),
                            sizeof(int),
                            py::format_descriptor<int>::format(),
                            1,
                            {labels->at(i)->info().dims().at(0)},
                            {sizeof(int) }));
            }
            return labels_list;
    }
            );
        m.def(
            "rocalGetBoundingBoxCords", [](RocalContext context)
    {
            rocalTensorList *boxes = rocalGetBoundingBoxCords(context);
            py::list boxes_list;
            py::array_t<double> boxes_array;
            for (int i = 0; i < boxes->size(); i++) {
                double *box_buffer = (double *)(boxes->at(i)->buffer());
                boxes_array = py::array(py::buffer_info(
                            (double *)(boxes->at(i)->buffer()),
                            sizeof(double),
                            py::format_descriptor<double>::format(),
                            1,
                            { boxes->at(i)->info().dims().at(0) * 4},
                            {sizeof(double) }));
                boxes_list.append(boxes_array);
            }
            return boxes_list;
    }
            );
        m.def(
            "rocalGetMatchedIndices", [](RocalContext context)
    {
            rocalTensorList *matches = rocalGetMatchedIndices(context);
            return py::array(py::buffer_info(
                            (int *)(matches->at(0)->buffer()),
                            sizeof(int),
                            py::format_descriptor<int>::format(),
                            1,
                            {matches->size() * 120087},
                            {sizeof(int) }));
    }, py::return_value_policy::reference);
        m.def(
            "rocalGetEncodedBoxesAndLables", [](RocalContext context,uint batch_size, uint num_anchors)
            {
                auto vec_pair_labels_boxes = rocalGetEncodedBoxesAndLables(context, batch_size * num_anchors);
                auto labels_buf_ptr = (int*)(vec_pair_labels_boxes[0]->at(0)->buffer());
                auto bboxes_buf_ptr = (float*)(vec_pair_labels_boxes[1]->at(0)->buffer());
                THROW("UNKNOWN DTYE")
                py::array_t<int> labels_array = py::array_t<int>(py::buffer_info(
                            labels_buf_ptr,
                            sizeof(int),
                            py::format_descriptor<int>::format(),
                            2,
                            {batch_size, num_anchors},
                            {num_anchors*sizeof(int), sizeof(int)}));

                py::array_t<float> bboxes_array = py::array_t<float>(py::buffer_info(
                            bboxes_buf_ptr,
                            sizeof(float),
                            py::format_descriptor<float>::format(),
                            1,
                            {batch_size * num_anchors * 4},
                            {sizeof(float)} ));

        return std::make_pair(labels_array, bboxes_array);
            }
        );
        // rocal_api_data_loaders.h
        m.def("ImageDecoder", &rocalJpegFileSource, "Reads file from the source given and decodes it according to the policy",
              py::return_value_policy::reference);
        m.def("ImageDecoderShard", &rocalJpegFileSourceSingleShard, "Reads file from the source given and decodes it according to the shard id and number of shards",
              py::return_value_policy::reference);
        m.def("FusedDecoderCropShard",&rocalFusedJpegCropSingleShard,"Reads file from the source and decodes them partially to output random crops",
            py::return_value_policy::reference);
        m.def("COCO_ImageDecoderShard",&rocalJpegCOCOFileSourceSingleShard,"Reads file from the source given and decodes it according to the shard id and number of shards",
            py::return_value_policy::reference);
        m.def("COCO_ImageDecoderSliceShard",&rocalJpegCOCOFileSourcePartialSingleShard,"Reads file from the source given and decodes it according to the policy",
            py::return_value_policy::reference);
        m.def("Audio_DecoderSliceShard",&rocalAudioFileSourceSingleShard,"Reads file from the source given and decodes it according to the policy",
            py::return_value_policy::reference);
        m.def("Audio_decoder",&rocalAudioFileSource,"Reads file from the source given and decodes it according to the policy",
            py::return_value_policy::reference);
        // rocal_api_augmentation.h
        // Audio Augmentations
        m.def("NormalDistribution", &rocalNormalDistribution, "Generates random numbers following a normal distribution",
            py::return_value_policy::reference);
        m.def("UniformDistribution", &rocalUniformDistribution, "Generates random numbers following a normal distribution",
            py::return_value_policy::reference);
        m.def("ToDecibels", &rocalToDecibels, "Converts to Decibals",
            py::return_value_policy::reference);
        m.def("PreEmphasisFilter", &rocalPreEmphasisFilter, "Applies preemphasis filter to the input data", 
            py::return_value_policy::reference);
        m.def("Spectrogram", &rocalSpectrogram, " Produces a spectrogram from a 1D signal (for example, audio)",
            py::return_value_policy::reference);
        m.def("MelFilterBank", &rocalMelFilterBank, "Converts a spectrogram to a mel spectrogram by applying a bank of triangular filters",
            py::return_value_policy::reference);
        m.def("audioSlice", &rocalSlice,"The slice can be specified by proving the start and end coordinates, or start coordinates and shape of the slice. Both coordinates and shapes can be provided in absolute or relative terms",
            py::return_value_policy::reference);
        m.def("audioNormalize", &rocalNormalize,"Normalizes the input by removing the mean and dividing by the standard deviation",
            py::return_value_policy::reference);
        m.def("NonSilentRegion", &rocalNonSilentRegion,"  Performs leading and trailing silence detection in an audio buffer",
            py::return_value_policy::reference);
        m.def("Pad", &rocalPad," Pads all samples with the fill_value in the specified axes to match the biggest extent in the batch for those axes or to match the minimum shape specified",
            py::return_value_policy::reference);
        m.def("TensorMulScalar", &rocalTensorMulScalar, "Multiplies a given Tensor Value with Scalar - Arithmetic Operation",
            py::return_value_policy::reference);
        m.def("TensorAddTensor", &rocalTensorAddTensor, "Multiplies a given Tensor Value with Scalar - Arithmetic Operation",
            py::return_value_policy::reference);
        m.def("Resample", &rocalResample, "Resamples the audio",
            py::return_value_policy::reference);
        // Image Augmentations
        m.def("Resize",&rocalResize, "Resizes the image ",py::return_value_policy::reference);
        m.def("ColorTwist",&rocalColorTwist, py::return_value_policy::reference);
        m.def("rocalResetLoaders", &rocalResetLoaders);
        m.def("Brightness", &rocalBrightness,
              py::return_value_policy::reference);
        m.def("CropMirrorNormalize",&rocalCropMirrorNormalize, py::return_value_policy::reference);
        m.def("ResizeMirrorNormalize",&rocalResizeMirrorNormalize, py::return_value_policy::reference);
        // m.def("Crop", &rocalCrop, py::return_value_policy::reference);
        m.def("CenterCropFixed", &rocalCropCenterFixed, py::return_value_policy::reference);
    }
}
