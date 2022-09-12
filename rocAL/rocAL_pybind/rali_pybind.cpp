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
namespace rali
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

    py::object wrapper(RocalContext context, py::array_t<unsigned char> array)
    {
        auto buf = array.request();
        unsigned char *ptr = (unsigned char *)buf.ptr;
        // call pure C++ function
        int status = rocalCopyToOutput(context, ptr, buf.size);
        return py::cast<py::none>(Py_None);
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

    PYBIND11_MODULE(rali_pybind, m)
    {
        m.doc() = "Python bindings for the C++ portions of ROCAL";
        // rocal_api.h
        m.def("rocalCreate", &rocalCreate, "Creates context with the arguments sent and returns it",
              py::return_value_policy::reference,
              py::arg("batch_size"),
              py::arg("affinity"),
              py::arg("gpu_id") = 0,
              py::arg("cpu_thread_count") = 1,
              py::arg("prefetch_queue_depth") = 3,
              py::arg("output_data_type") = 0);
        m.def("rocalVerify", &rocalVerify);
        m.def("rocalRun", &rocalRun);
        m.def("rocalRelease", &rocalRelease);
        // rocal_api_types.h
        py::class_<TimingInfo>(m, "TimingInfo")
            .def_readwrite("load_time", &TimingInfo::load_time)
            .def_readwrite("decode_time", &TimingInfo::decode_time)
            .def_readwrite("process_time", &TimingInfo::process_time)
            .def_readwrite("transfer_time", &TimingInfo::transfer_time);
        py::class_<rocALTensor>(m, "rocALTensor")
            .def(
                "batch_height",
                [](rocALTensor &output_tensor) {
                    return output_tensor.info().max_dims().at(1);
                },
                R"code(
                Returns a tensor buffer's height.
                )code")
            .def(
                "batch_width",
                [](rocALTensor &output_tensor) {
                    return output_tensor.info().max_dims().at(0);
                },
                R"code(
                Returns a tensor buffer's width.
                )code")
            .def(
                "batch_size",
                [](rocALTensor &output_tensor) {
                    return output_tensor.info().dims().at(0);
                },
                R"code(
                Returns a tensor batch size.
                )code")
            .def(
                "color_format",
                [](rocALTensor &output_tensor) {
                    if ((output_tensor.info().color_format() == RocalColorFormat::RGB24) || (output_tensor.info().color_format() == RocalColorFormat::BGR24))
                        return 3;
                    else
                        return 1;
                },
                R"code(
                Returns a tensor batch size.
                )code")
            .def(
                "copy_data", [](rocALTensor &output_tensor, py::object p) {
                    std::cerr << "COOPY DATA PY";
                    auto ptr = ctypes_void_ptr(p);
                    output_tensor.copy_data(ptr, 0);
                })
            .def("copy_data_numpy", [](rocALTensor &output_tensor, py::array_t<unsigned char> array) {
                auto buf = array.request();
                unsigned char *ptr = (unsigned char *)buf.ptr;
                std::cerr << "COOPpppY DATA PY";
                output_tensor.copy_data((void *)ptr, 0);
            })
            .def(
                "at", [](rocALTensor &output_tensor, uint idx) {
                    uint h = output_tensor.info().max_dims().at(1);
                    uint w = output_tensor.info().max_dims().at(0);

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
                Returns a rocAL tensor at given position `i` in the rocALTensorlist.
                )code",
                py::keep_alive<0, 1>());

        // .def_readwrite("swap_handle",&rocALTensor::swap_handle);
        py::class_<rocALTensorList>(m, "rocALTensorList")
            .def(
                "__getitem__",
                [](rocALTensorList &output_tensor_list, uint idx)
                {
                    return output_tensor_list.at(idx);
                },
                R"code(
                Returns a tensor at given position in the list.
                )code")

            .def("at",
                [](rocALTensorList &output_tensor_list, uint idx)
                {
                    uint h = output_tensor_list.at(idx)->info().max_dims().at(1);
                    uint w = output_tensor_list.at(idx)->info().max_dims().at(0);

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
                Returns a rocAL tensor at given position `i` in the rocALTensorlist.
                )code",
                py::keep_alive<0, 1>());
        py::class_<rocALTensorInfo>(m, "rocALTensorInfo");

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
        // rocal_api_info.h
        m.def("getRemainingImages", &rocalGetRemainingImages);
        m.def("isEmpty", &rocalIsEmpty);
        m.def("getStatus", rocalGetStatus);
        m.def("rocalGetErrorMessage", &rocalGetErrorMessage);
        m.def("getTimingInfo", &rocalGetTimingInfo);
        m.def("setOutputImages", &rocalSetOutputs);
        m.def("labelReader", &rocalCreateLabelReader, py::return_value_policy::reference);
        m.def("COCOReader", &rocalCreateCOCOReader, py::return_value_policy::reference);
        // rocal_api_meta_data.h
        m.def("TFReader",&rocalCreateTFReader);
        m.def("TFReaderDetection",&rocalCreateTFReaderDetection);

        m.def("RandomBBoxCrop", &rocalRandomBBoxCrop);
        m.def("BoxEncoder",&rocalBoxEncoder);
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
            // call pure C++ function
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
        m.def("CreateFloatRand", &rocalCreateFloatRand);
        m.def("CreateIntParameter", &rocalCreateIntParameter);
        m.def("CreateFloatParameter", &rocalCreateFloatParameter);
        m.def("UpdateIntParameter", &rocalUpdateIntParameter);
        m.def("UpdateFloatParameter", &rocalUpdateFloatParameter);
        m.def("GetIntValue", &rocalGetIntValue);
        m.def("GetFloatValue", &rocalGetFloatValue);
        // rocal_api_data_transfer.h
        // m.def("rocalGetOutputTensors",&rocalGetOutputTensors, return_value_policy::reference);
        m.def("rocalGetOutputTensors", [](RocalContext context)
              {
            rocALTensorList * tl = rocalGetOutputTensors(context);
            py::list list;
            unsigned int size_of_tensor_list = tl->size();
            for (uint i =0; i< size_of_tensor_list; i++)
                list.append(tl->at(i));
            return list; });
        m.def(
            "rocalGetImageLabels", [](RocalContext context)
    {
            rocALTensorList *labels = rocalGetImageLabels(context);
            // std::cerr<<"LABELS SIZE ::"<<labels->size();
            // for (int i = 0; i < labels->size(); i++) {
            //     int *labels_buffer = (int *)(labels->at(i)->buffer());
            //     std::cerr << ">>>>> LABELS : " << labels_buffer[0] << "\t";
            // }
            return py::array(py::buffer_info(
                            (int *)(labels->at(0)->buffer()),
                            sizeof(int),
                            py::format_descriptor<int>::format(),
                            1,
                            {labels->size()},
                            {sizeof(int) }));
    }
            );

        m.def("rocalGetBoundingBoxLabel", [](RocalContext context)
              {
            std::cerr<<"***************************rocalGetBoundingBoxLabel**********************************";
            rocALTensorList * bbox_labels = rocalGetBoundingBoxLabel(context);
            py::list list,list2;
            for(int i = 0; i < bbox_labels->size(); i++)
                {
                    auto labels_buffer = (int *)(bbox_labels->at(i)->buffer());
                    // auto bbox_buffer = (float *)(bbox_coords->at(i)->buffer());
                    std::cerr << "\n>>>>> BBOX LABELS : ";
                    int label_size = bbox_labels->at(i)->info().dims().at(0);
                    std::cerr<<"label_size "<<bbox_labels;
                    py::array_t<int> labels_array = py::array_t<int>(py::buffer_info(
                            labels_buffer,
                            sizeof(int),
                            py::format_descriptor<int>::format(),
                            1,
                            {label_size},
                            {sizeof(int)}));
                    std::cerr<<"***********************"<<list<<"\n";
                    list.append(labels_array);
                    // list = list +labels_array;
                    // list.merge(labels_array);

                    std::cerr<<"***********************"<<list<<"\n";

                }
                std::cerr<<"list"<<list;
                return list;
            });

        m.def("rocalGetBoundingBoxCords", [](RocalContext context)
              {
            std::cerr<<"***************************rocalGetBoundingBoxCords**********************************";
            rocALTensorList* bbox_coords = rocalGetBoundingBoxCords(context);
            py::list list;
            for(int i = 0; i < bbox_coords->size(); i++)
                {
                    auto bbox_buffer = (float *)(bbox_coords->at(i)->buffer());
                    std::cerr << "\n>>>>> BBOXX : ";
                    int label_size = bbox_coords->at(i)->info().dims().at(0);
                    std::cerr<<"label_size "<<bbox_coords;
                    py::array_t<float> bboxes_array = py::array_t<float>(py::buffer_info(
                            bbox_buffer,
                            sizeof(float),
                            py::format_descriptor<float>::format(),
                            1,
                            {label_size*4},
                            {sizeof(float)}));
                        
                    list.append(bboxes_array);
                    std::cerr<<"bboxes_array",bboxes_array;
                }
                std::cerr<<"list"<<list;
                return list;
            });
        // m.def(
        //     "copy_data_ptr", [](RocalContext context, py::object p)
        // {
        // auto ptr = ctypes_void_ptr(p);
        // RocalTensorList output_tensor_list = rocalGetOutputTensors(context);
        // // ptr = output_tensor_list->at(0)->buffer();

        // rocALTensor::copy_data((unsigned char *) ptr, 0);
        // // for (uint i =0; i<10; i++)
        // // {
        // //     std::cerr<<"\n TEMP ::"<< (float) (unsigned char *) ptr[i];
        // // }
        // // std::exit(0);
        // return py::reinterpret_borrow<py::object>(PyLong_FromVoidPtr(ptr));
        // }
        //     );
        m.def(
            "rocalGetEncodedBoxesAndLables", [](RocalContext context,uint batch_size, uint num_anchors)
            {
                auto vec_pair_labels_boxes = rocalGetEncodedBoxesAndLables(context, batch_size * num_anchors);
                auto labels_buf_ptr = (int*)(vec_pair_labels_boxes[0]->at(0)->buffer());
                auto bboxes_buf_ptr = (float*)(vec_pair_labels_boxes[1]->at(0)->buffer());

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
              py::return_value_policy::reference,
              py::arg("context"),
              py::arg("source_path"),
              py::arg("color_format"),
              py::arg("num_threads"),
              py::arg("is_output") = false,
              py::arg("shuffle") = false,
              py::arg("loop") = false,
              py::arg("decode_size_policy") = ROCAL_USE_MOST_FREQUENT_SIZE,
              py::arg("max_width") = 0,
              py::arg("max_height") = 0,
              py::arg("dec_type") = 0);
        m.def("ImageDecoderShard", &rocalJpegFileSourceSingleShard, "Reads file from the source given and decodes it according to the shard id and number of shards",
              py::return_value_policy::reference,
              py::arg("context"),
              py::arg("source_path"),
              py::arg("color_format"),
              py::arg("shard_id"),
              py::arg("shard_count"),
              py::arg("is_output") = false,
              py::arg("shuffle") = false,
              py::arg("loop") = false,
              py::arg("decode_size_policy") = ROCAL_USE_MOST_FREQUENT_SIZE,
              py::arg("max_width") = 0,
              py::arg("max_height") = 0);
        m.def("FusedDecoderCropShard",&rocalFusedJpegCropSingleShard,"Reads file from the source and decodes them partially to output random crops",
            py::return_value_policy::reference);
        m.def("COCO_ImageDecoderSliceShard",&rocalJpegCOCOFileSourcePartialSingleShard,"Reads file from the source given and decodes it according to the policy");
        m.def("TF_ImageDecoder",&rocalJpegTFRecordSource,"Reads file from the source given and decodes it according to the policy only for TFRecords",
            py::return_value_policy::reference,
            py::arg("p_context"),
            py::arg("source_path"),
            py::arg("rocal_color_format"),
            py::arg("internal_shard_count"),
            py::arg("is_output"),
            py::arg("user_key_for_encoded"),
            py::arg("user_key_for_filename"),
            py::arg("shuffle") = false,
            py::arg("loop") = false,
            py::arg("decode_size_policy") = ROCAL_USE_MOST_FREQUENT_SIZE,
            py::arg("max_width") = 0,
            py::arg("max_height") = 0);
        m.def("Resize",&rocalResize, "Resizes the image ",py::return_value_policy::reference);
        m.def("ColorTwist",&rocalColorTwist, py::return_value_policy::reference);
        m.def("rocalResetLoaders", &rocalResetLoaders);
        // rocal_api_augmentation.h
        // m.def("Brightness", &rocalBrightness,
        //       py::return_value_policy::reference,
        //       py::arg("context"),
        //       py::arg("input"),
        //       py::arg("is_output"),
        //       py::arg("alpha") = NULL,
        //       py::arg("beta") = NULL);
        m.def("Brightness", &rocalBrightness, py::return_value_policy::reference);
        m.def("Gamma", &rocalGamma, py::return_value_policy::reference);
        m.def("Blur", &rocalBlur, py::return_value_policy::reference);
        m.def("Hue", &rocalHue, py::return_value_policy::reference);
        m.def("Contrast", &rocalContrast, py::return_value_policy::reference);
        m.def("flip", &rocalFlip, py::return_value_policy::reference);
        m.def("Jitter", &rocalJitter, py::return_value_policy::reference);
        m.def("pixelate", &rocalPixelate, py::return_value_policy::reference);
        m.def("Rain", &rocalRain, py::return_value_policy::reference);
        m.def("Rotate", &rocalRotate, py::return_value_policy::reference);
        m.def("Saturation", &rocalSaturation, py::return_value_policy::reference);
        m.def("warp_affine", &rocalWarpAffine, py::return_value_policy::reference);
        m.def("Vignette", &rocalVignette, py::return_value_policy::reference);
        m.def("ColorTemp", &rocalBlur, py::return_value_policy::reference);
        m.def("snp_noise", &rocalNoise, py::return_value_policy::reference);

        // m.def("Hue", &rocalHue, py::return_value_policy::reference);

        m.def("CropMirrorNormalize",&rocalCropMirrorNormalize, py::return_value_policy::reference);
        // m.def("Crop", &rocalCrop, py::return_value_policy::reference);
        m.def("ResizeShorter", &rocalResizeShorter, py::return_value_policy::reference);
        m.def("CenterCropFixed", &rocalCropCenterFixed, py::return_value_policy::reference);


    }
}
