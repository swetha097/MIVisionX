/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.

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

#include "internal_publishKernels.h"
#include <Python.h>
#include <iomanip>
#include <sstream>
#include <vector>
#include <fstream>

void castData(int type, void* data, PyObject* pItem, int index) {
    std::string result;
    switch(type) {
        case 0:
            static_cast<vx_float32*>(data)[index] = (vx_float32)PyFloat_AsDouble(pItem);
            break;
        case 1:
            #if defined(AMD_FP16_SUPPORT)
                static_cast<vx_float16*>(data)[index] = (vx_float16)PyFloat_AsDouble(pItem);
            #else
                std::cerr<<"\n FLOAT16 type tensor not supported";
                return;
            #endif
            break;
        case 2:
            static_cast<uint8_t*>(data)[index] = static_cast<int>((uint8_t)PyLong_AsLong(pItem));
            break;
        case 3:
            static_cast<vx_int8*>(data)[index] = (vx_int8)PyLong_AsLong(pItem);
            break;
        case 4:
            static_cast<vx_uint32*>(data)[index] = (vx_uint32)PyLong_AsLong(pItem);
            break;
        case 5:
            static_cast<vx_int32*>(data)[index] = (vx_int32)PyLong_AsLong(pItem);
            break;
        // Handle more data types as needed
        default:
            break;
    }
}

std::string byteToHexString(char* bytes, size_t length) {
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for (size_t i = 0; i < length; ++i) {
        ss << std::setw(2) << static_cast<unsigned>(bytes[i]);
    }
    return ss.str();
}

struct ExternalSourceLocalData {
    vxRppHandle *handle;
    vx_uint32 deviceType;
    RppPtr_t pSrc;
    RppPtr_t pDst;
    RpptDescPtr pSrcDesc;
    RpptDescPtr pDstDesc;
    RpptROI *pSrcRoi;
    RpptRoiType roiType;
    vxTensorLayout inputLayout;
    vxTensorLayout outputLayout;
    vx_char *pFilePath;
    vx_uint32 dtype;
    size_t inputTensorDims[RPP_MAX_TENSOR_DIMS];
    size_t ouputTensorDims[RPP_MAX_TENSOR_DIMS];
    size_t filePathSize;
};

static vx_status VX_CALLBACK refreshExternalSource(vx_node node, const vx_reference *parameters, vx_uint32 num, ExternalSourceLocalData *data) {
    vx_status status = VX_SUCCESS;
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[3], 0, data->filePathSize, sizeof(vx_char), data->pFilePath, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    data->pFilePath[data->filePathSize] = '\0';

    void *roi_tensor_ptr;
    if (data->deviceType == AGO_TARGET_AFFINITY_GPU) {
#if ENABLE_OPENCL
        return VX_ERROR_NOT_IMPLEMENTED;
#elif ENABLE_HIP
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HIP, &roi_tensor_ptr, sizeof(roi_tensor_ptr)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HIP, &data->pSrc, sizeof(data->pSrc)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HIP, &data->pDst, sizeof(data->pDst)));
#endif
    } else if (data->deviceType == AGO_TARGET_AFFINITY_CPU) {
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HOST, &roi_tensor_ptr, sizeof(roi_tensor_ptr)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc, sizeof(data->pSrc)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HOST, &data->pDst, sizeof(data->pDst)));
    }
    data->pSrcRoi = reinterpret_cast<RpptROI *>(roi_tensor_ptr);
    if (data->inputLayout == vxTensorLayout::VX_NFHWC || data->inputLayout == vxTensorLayout::VX_NFCHW) {
        unsigned num_of_frames = data->inputTensorDims[1]; // Num of frames 'F'
        for (int n = data->inputTensorDims[0] - 1; n >= 0; n--) {
            unsigned index = n * num_of_frames;
            for (unsigned f = 0; f < num_of_frames; f++) {
                data->pSrcRoi[index + f].xywhROI = data->pSrcRoi[n].xywhROI;
            }
        }
    }
    return status;
}

static vx_status VX_CALLBACK validateExternalSource(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[]) {
    vx_status status = VX_SUCCESS;
    vx_enum scalar_type;
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[5], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Parameter: #8 type=%d (must be size)\n", scalar_type);

    // Check for input tensor
    size_t num_tensor_dims;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    if(num_tensor_dims < 1) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: ExternalSource: tensor: #0 dimensions=%lu (must be greater than or equal to 1)\n", num_tensor_dims);

    // Check for output tensor
    vx_uint8 tensor_fixed_point_position;
    size_t tensor_dims[RPP_MAX_TENSOR_DIMS];
    vx_enum tensor_dtype;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    if(num_tensor_dims < 1) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: ExternalSource: tensor: #2 dimensions=%lu (must be greater than or equal to 1)\n", num_tensor_dims);
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, &tensor_dims, sizeof(tensor_dims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DATA_TYPE, &tensor_dtype, sizeof(tensor_dtype)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_FIXED_POINT_POSITION, &tensor_fixed_point_position, sizeof(tensor_fixed_point_position)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_DIMS, &tensor_dims, sizeof(tensor_dims)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_DATA_TYPE, &tensor_dtype, sizeof(tensor_dtype)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_FIXED_POINT_POSITION, &tensor_fixed_point_position, sizeof(tensor_fixed_point_position)));
    return status;
}

static vx_status VX_CALLBACK processExternalSource(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    RppStatus rpp_status = RPP_SUCCESS;
    vx_status return_status = VX_SUCCESS;
    ExternalSourceLocalData *data = NULL;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    refreshExternalSource(node, parameters, num, data);
    std::ifstream file(data->pFilePath, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file\n";
        return 1;
    }

    // Read the file contents into a vector
    std::vector<char> buffer(std::istreambuf_iterator<char>(file), {});

    // Convert the vector to a Python bytes object
    PyObject* pBytes = PyBytes_FromStringAndSize(buffer.data(), buffer.size());

    // Deserialize the Python bytes object using Dill
    PyObject* pModule = PyImport_ImportModule("dill");
    if (pModule != NULL) {
        PyObject* pFunc = PyObject_GetAttrString(pModule, "loads");
        if (pFunc != NULL && PyCallable_Check(pFunc)) {
            PyObject* pResult_Tuple = PyObject_CallFunctionObjArgs(pFunc, pBytes, NULL);

            // Check if the deserialization was successful
            if (pResult_Tuple != NULL) {
                // Access individual elements of the tuple
                if (PyTuple_Check(pResult_Tuple)) {
                    // You can extract and process each element of the tuple here
                    // For example, print the function name and its qualified name
                    PyObject* basic_def = PyTuple_GetItem(pResult_Tuple, 1);
                    PyObject* fun_context = PyTuple_GetItem(pResult_Tuple, 2);
                    PyObject* set_funcion_state = PyTuple_GetItem(pResult_Tuple, 3);

                    // Extract information from basic_def
                    std::string name = PyUnicode_AsUTF8(PyTuple_GetItem(basic_def, 0));
                    std::string qualname = PyUnicode_AsUTF8(PyTuple_GetItem(basic_def, 1));
                    PyObject* code_obj = PyTuple_GetItem(basic_def, 2);
                    PyObject* closure = PyTuple_GetItem(basic_def, 3);

                    // Create a new Python function
                    PyObject* types_module = PyImport_ImportModule("types");
                    PyObject* FunctionType = PyObject_GetAttrString(types_module, "FunctionType");

                    const char* builtins_module_name = PY_MAJOR_VERSION == 2 ? "__builtin__" : "builtins";
                    PyObject* builtins_module = PyImport_ImportModule(builtins_module_name);

                    PyObject* global_scope = PyDict_New();
                    PyDict_SetItemString(global_scope, "__builtins__", builtins_module);

                    PyObject* marshal_module = PyImport_ImportModule("marshal");
                    PyObject* loads_func = PyObject_GetAttrString(marshal_module, "loads");
                    PyObject* code = PyObject_CallFunction(loads_func, "N", code_obj);

                    PyObject* fun_args = PyTuple_Pack(4, code, global_scope, PyUnicode_FromString(name.c_str()), closure);
                    PyObject* fun = PyObject_CallObject(FunctionType, fun_args);

                    // Set the function state
                    PyObject* result_set_state = PyObject_CallFunction(set_funcion_state, "OO", fun, fun_context);

                    // Execute the function
                    PyObject* pResult = PyObject_CallFunctionObjArgs(fun, PyLong_FromLong(data->pSrcDesc->n), NULL);
                    if (pResult != NULL && PyList_Check(pResult)) {
                        int listSize = PyList_Size(pResult);
                        for (int i = 0; i < listSize; i++) {
                            PyObject* pItem = PyList_GetItem(pResult, i);
                            // Handle each item as needed (e.g., castData)
                            castData(data->dtype, data->pDst, pItem, i);
                        }
                    }
                } else {
                    std::cerr << "Error: Deserialized object is not a tuple or has incorrect size\n";
                }

                Py_DECREF(pResult_Tuple);
            } else {
                PyErr_Print();
            }

            Py_DECREF(pFunc);
        } else {
            PyErr_Print();
        }

        Py_DECREF(pModule);
    } else {
        PyErr_Print();
    }

    return return_status;
}

static vx_status VX_CALLBACK initializeExternalSource(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    ExternalSourceLocalData *data = new ExternalSourceLocalData;
    memset(data, 0, sizeof(ExternalSourceLocalData));
    vx_enum input_tensor_dtype, output_tensor_dtype;
    vx_int32 roi_type, input_layout, output_layout;
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[5], &data->deviceType, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    data->roiType = static_cast<RpptRoiType>(roi_type);
    data->inputLayout = static_cast<vxTensorLayout>(input_layout);
    data->outputLayout = static_cast<vxTensorLayout>(output_layout);

    // Querying for input tensor
    data->pSrcDesc = new RpptDesc;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &data->pSrcDesc->numDims, sizeof(data->pSrcDesc->numDims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, &data->inputTensorDims, sizeof(vx_size) * data->pSrcDesc->numDims));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &input_tensor_dtype, sizeof(input_tensor_dtype)));
    data->pSrcDesc->dataType = getRpptDataType(input_tensor_dtype);
    data->pSrcDesc->offsetInBytes = 0;

    // Querying for output tensor
    data->pDstDesc = new RpptDesc;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_NUMBER_OF_DIMS, &data->pDstDesc->numDims, sizeof(data->pDstDesc->numDims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, &data->ouputTensorDims, sizeof(vx_size) * data->pDstDesc->numDims));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DATA_TYPE, &output_tensor_dtype, sizeof(output_tensor_dtype)));
    data->pDstDesc->dataType = getRpptDataType(output_tensor_dtype);
    data->pDstDesc->offsetInBytes = 0;
    data->pSrcDesc->n = data->ouputTensorDims[0];
    STATUS_ERROR_CHECK(vxQueryArray((vx_array)parameters[3], VX_ARRAY_CAPACITY, &data->filePathSize, sizeof(data->filePathSize)));
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[4], &data->dtype, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    data->pFilePath = new char[data->filePathSize + 1];

    refreshExternalSource(node, parameters, num, data);
    STATUS_ERROR_CHECK(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK uninitializeExternalSource(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    ExternalSourceLocalData *data;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    std::remove(data->pFilePath);
    delete[] data->pFilePath;
    delete data->pSrcDesc;
    delete data->pDstDesc;
    delete data;
    return VX_SUCCESS;
}

//! \brief The kernel target support callback.
// TODO::currently the node is setting the same affinity as context. This needs to change when we have hybrid modes in the same graph
static vx_status VX_CALLBACK query_target_support(vx_graph graph, vx_node node,
                                                  vx_bool use_opencl_1_2,              // [input]  false: OpenCL driver is 2.0+; true: OpenCL driver is 1.2
                                                  vx_uint32 &supported_target_affinity // [output] must be set to AGO_TARGET_AFFINITY_CPU or AGO_TARGET_AFFINITY_GPU or (AGO_TARGET_AFFINITY_CPU | AGO_TARGET_AFFINITY_GPU)
) {
    vx_context context = vxGetContext((vx_reference)graph);
    AgoTargetAffinityInfo affinity;
    vxQueryContext(context, VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY, &affinity, sizeof(affinity));
    if (affinity.device_type == AGO_TARGET_AFFINITY_GPU)
        supported_target_affinity = AGO_TARGET_AFFINITY_GPU;
    else
        supported_target_affinity = AGO_TARGET_AFFINITY_CPU;

    return VX_SUCCESS;
}

vx_status ExternalSource_Register(vx_context context) {
    vx_status status = VX_SUCCESS;
    // Add kernel to the context with callbacks
    vx_kernel kernel = vxAddUserKernel(context, "org.rpp.ExternalSource",
                                       VX_KERNEL_EXTERNALSOURCE,
                                       processExternalSource,
                                       6,
                                       validateExternalSource,
                                       initializeExternalSource,
                                       uninitializeExternalSource);
    ERROR_CHECK_OBJECT(kernel);
    AgoTargetAffinityInfo affinity;
    vxQueryContext(context, VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY, &affinity, sizeof(affinity));
#if ENABLE_HIP
    vx_bool enableBufferAccess = vx_true_e;
    if (affinity.device_type == AGO_TARGET_AFFINITY_GPU)
        STATUS_ERROR_CHECK(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_GPU_BUFFER_ACCESS_ENABLE, &enableBufferAccess, sizeof(enableBufferAccess)));
#else
    vx_bool enableBufferAccess = vx_false_e;
#endif
    amd_kernel_query_target_support_f query_target_support_f = query_target_support;

    if (kernel) {
        STATUS_ERROR_CHECK(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 2, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxFinalizeKernel(kernel));
    }
    if (status != VX_SUCCESS) {
    exit:
        vxRemoveKernel(kernel);
        return VX_FAILURE;
    }

    return status;
}
