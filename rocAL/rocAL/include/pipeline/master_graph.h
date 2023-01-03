/*
Copyright (c) 2019 - 2022 Advanced Micro Devices, Inc. All rights reserved.

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

#pragma once
#include <memory>
#include <list>
#include <variant>
#include <map>
#include "graph.h"
#include "ring_buffer.h"
#include "timing_debug.h"
#include "node.h"
#include "node_image_loader.h"
#include "node_image_loader_single_shard.h"
#include "node_audio_loader.h"
#include "node_audio_loader_single_shard.h"
#include "node_fused_jpeg_crop.h"
#include "node_fused_jpeg_crop_single_shard.h"
#include "node_video_loader.h"
#include "node_video_loader_single_shard.h"
#include "node_cifar10_loader.h"
#include "meta_data_reader.h"
#include "meta_data_graph.h"
#if ENABLE_HIP
#include "device_manager_hip.h"
#include "box_encoder_hip.h"
#endif
#include "randombboxcrop_meta_data_reader.h"
#define MAX_STRING_LENGTH 100
#define MAX_OBJECTS 50 // Max number of objects/image in COCO dataset is 93 
#define BBOX_COUNT 4
#define MAX_NUM_ANCHORS 8732
#define MAX_MASK_BUFFER 10000

class MasterGraph
{
public:
    enum class Status { OK = 0,  NOT_RUNNING = 1, NO_MORE_DATA = 2, NOT_IMPLEMENTED = 3, INVALID_ARGUMENTS };
    MasterGraph(size_t batch_size, RocalAffinity affinity, int gpu_id, size_t prefetch_queue_depth, RocalTensorDataType output_tensor_data_type);
    ~MasterGraph();
    Status reset();
    size_t remaining_count();
    MasterGraph::Status copy_output(std::vector<void *> &out_ptr);
    Status copy_output(void* out_ptr, size_t out_size);
    std::vector<uint32_t> output_resize_width();
    std::vector<uint32_t> output_resize_height();
    std::vector<size_t> tensor_output_byte_size();
    void sequence_start_frame_number(std::vector<size_t> &sequence_start_framenum); // Returns the starting frame number of the sequences
    void sequence_frame_timestamps(std::vector<std::vector<float>> &sequence_frame_timestamp); // Returns the timestamps of the frames in the sequences
    Status build();
    Status run();
    Timing timing();
    RocalMemType mem_type();
    void release();
    template <typename T>
    std::shared_ptr<T> add_node(const std::vector<rocalTensor *> &input, const std::vector<rocalTensor *> &output);
    vx_context get_vx_context() { return _context; }
    template <typename T, typename M> std::shared_ptr<T> meta_add_node(std::shared_ptr<M> node);
    rocalTensor *create_tensor(const rocalTensorInfo &info, bool is_output);
    rocalTensor *create_loader_output_tensor(const rocalTensorInfo &info);
    rocalTensor *create_output_tensor(const rocalTensorInfo &info);
    rocalTensorList * get_output_tensors();
    std::vector<rocalTensorList *> create_label_reader(const char *source_path, MetaDataReaderType reader_type);
    std::vector<rocalTensorList *> create_file_list_label_reader(const char *source_path, const char *file_list_path, MetaDataReaderType reader_type);
    std::vector<rocalTensorList *> create_video_label_reader(const char *source_path, MetaDataReaderType reader_type, unsigned sequence_length, unsigned frame_step, unsigned frame_stride, bool file_list_frame_num = true);
    std::vector<rocalTensorList *> create_coco_meta_data_reader(const char *source_path, bool is_output, bool mask, MetaDataReaderType reader_type, MetaDataType label_type, bool is_box_encoder = false);
    std::vector<rocalTensorList *> create_tf_record_meta_data_reader(const char *source_path, MetaDataReaderType reader_type,  MetaDataType label_type, const std::map<std::string, std::string> feature_key_map);
    std::vector<rocalTensorList *> create_caffe_lmdb_record_meta_data_reader(const char *source_path, MetaDataReaderType reader_type,  MetaDataType label_type);
    std::vector<rocalTensorList *> create_caffe2_lmdb_record_meta_data_reader(const char *source_path, MetaDataReaderType reader_type,  MetaDataType label_type);
    std::vector<rocalTensorList *> create_cifar10_label_reader(const char *source_path, const char *file_prefix);
    std::vector<rocalTensorList *> create_mxnet_label_reader(const char *source_path, bool is_output);

    void box_encoder(std::vector<float> &anchors, float criteria, const std::vector<float> &means, const std::vector<float> &stds, bool offset, float scale);
    void create_randombboxcrop_reader(RandomBBoxCrop_MetaDataReaderType reader_type, RandomBBoxCrop_MetaDataType label_type, bool all_boxes_overlap, bool no_crop, FloatParam* aspect_ratio, bool has_shape, int crop_width, int crop_height, int num_attempts, FloatParam* scaling, int total_num_attempts, int64_t seed=0);
    const std::pair<ImageNameBatch,pMetaDataBatch>& meta_data();
    rocalTensorList * labels_meta_data();
    rocalTensorList * bbox_labels_meta_data();
    rocalTensorList * bbox_meta_data();
    rocalTensorList * mask_meta_data();
    ImgSizes& get_image_sizes();

    void set_loop(bool val) { _loop = val; }
    void set_output(rocalTensor* output_image);
    bool empty() { return (remaining_count() < (_is_sequence_reader_output ? _sequence_batch_size : _user_batch_size)); }
    size_t internal_batch_size() { return _internal_batch_size; }
    size_t sequence_batch_size() { return _sequence_batch_size; }
    std::shared_ptr<MetaDataGraph> meta_data_graph() { return _meta_data_graph; }
    std::shared_ptr<MetaDataReader> meta_data_reader() { return _meta_data_reader; }
    bool is_random_bbox_crop() {return _is_random_bbox_crop; }
    bool is_segmentation() { return _is_segmentation; };
    std::vector<rocalTensorList *> get_bbox_encoded_buffers(size_t num_encoded_boxes);
    size_t bounding_box_batch_count(int* buf, pMetaDataBatch meta_data_batch);
    bool is_sequence_reader_output() {return _is_sequence_reader_output; }
    void set_sequence_reader_output() { _is_sequence_reader_output = true; }
    void set_sequence_batch_size(size_t sequence_length) { _sequence_batch_size = _user_batch_size * sequence_length; }
    void set_sequence_batch_ratio() { _sequence_batch_ratio = _sequence_batch_size / _internal_batch_size; }
private:
    Status update_node_parameters();
    Status allocate_output_tensor();
    Status deallocate_output_tensor();
    void create_single_graph();
    void start_processing();
    void stop_processing();
    void output_routine();
    void decrease_image_count();
    bool processing_on_device_ocl() { return _output_tensor_info.mem_type() == RocalMemType::OCL; };
    bool processing_on_device_hip() { return _output_tensor_info.mem_type() == RocalMemType::HIP; };
    /// notify_user_thread() is called when the internal processing thread is done with processing all available images
    void notify_user_thread();
    /// no_more_processed_data() is logically linked to the notify_user_thread() and is used to tell the user they've already consumed all the processed images
    bool no_more_processed_data();
    RingBuffer _ring_buffer;//!< The queue that keeps the images that have benn processed by the internal thread (_output_thread) asynchronous to the user's thread
    MetaDataBatch* _augmented_meta_data = nullptr;//!< The output of the meta_data_graph,
    CropCordBatch* _random_bbox_crop_cords_data = nullptr;
    std::thread _output_thread;
    rocalTensorInfo _output_tensor_info;
    rocalTensorList _internal_tensor_list;
    rocalTensorList _output_tensor_list;    //!< Keeps a list of ovx tensors that are used to store the augmented outputs (there is an augmentation output batch per element in the list)
    std::list<rocalTensor*> _internal_tensors;  //!< Keeps all the ovx tensors (virtual/non-virtual) either intermediate tensors, or input tensors that feed the graph
    std::list<std::shared_ptr<Node>> _nodes;
    std::list<std::shared_ptr<Node>> _root_nodes;
    std::list<std::shared_ptr<Node>> _meta_data_nodes;//!< List of nodes where meta data has to be updated after augmentation
    std::map<rocalTensor*, std::shared_ptr<Node>> _tensor_map;

    // Output tensorList for metadata
    std::vector<rocalTensorList *> _metadata_output_tensor_list;
    rocalTensorList _labels_tensor_list;
    rocalTensorList _bbox_tensor_list;
    rocalTensorList _mask_tensor_list;
    std::vector<std::vector<unsigned>> _labels_tensor_dims;
    std::vector<std::vector<unsigned>> _bbox_tensor_dims;
    std::vector<std::vector<unsigned>> _mask_tensor_dims;

    std::vector<size_t> _meta_data_buffer_size;

#if ENABLE_HIP
    void * _output_tensor;//!< In the GPU processing case , is used to convert the U8 samples to float32 before they are being transfered back to host
    DeviceManagerHip   _device;//!< Keeps the device related constructs needed for running on GPU
#else
    void* _output_tensor;//!< In the GPU processing case , is used to convert the U8 samples to float32 before they are being transfered back to host
    DeviceManager   _device;//!< Keeps the device related constructs needed for running on GPU
#endif
    std::shared_ptr<Graph> _graph = nullptr;
    RocalAffinity _affinity;
    const int _gpu_id;//!< Defines the device id used for processing
    pLoaderModule _loader_module; //!< Keeps the loader module used to feed the input the images of the graph
    TimingDBG _convert_time, _process_time, _bencode_time;
    const size_t _user_batch_size;//!< Batch size provided by the user
    vx_context _context;
    const RocalMemType _mem_type;//!< Is set according to the _affinity, if GPU, is set to CL, otherwise host
    std::shared_ptr<MetaDataReader> _meta_data_reader = nullptr;
    std::shared_ptr<MetaDataGraph> _meta_data_graph = nullptr;
    std::shared_ptr<RandomBBoxCrop_MetaDataReader> _randombboxcrop_meta_data_reader = nullptr;
    bool _first_run = true;
    bool _processing;//!< Indicates if internal processing thread should keep processing or not
    const static unsigned OUTPUT_RING_BUFFER_DEPTH = 3;
    const static unsigned SAMPLE_SIZE = sizeof(vx_float32); // unsigned char
    int _remaining_count;//!< Keeps the count of remaining images yet to be processed for the user,
    bool _loop;//!< Indicates if user wants to indefinitely loops through images or not
    static size_t compute_optimum_internal_batch_size(size_t user_batch_size, RocalAffinity affinity);
    const size_t _internal_batch_size;//!< In the host processing case , internal batch size can be different than _user_batch_size. This batch size used internally throughout.
    const size_t _user_to_internal_batch_ratio;
    size_t _prefetch_queue_depth;
    bool _output_routine_finished_processing = false;
    const RocalTensorDataType _out_data_type;
    bool _is_random_bbox_crop = false;
    bool _is_segmentation = false;
    std::vector<std::vector<uint32_t>> _resize_width;
    std::vector<std::vector<uint32_t>> _resize_height;
    // box encoder variables
    bool _is_box_encoder = false; //bool variable to set the box encoder
    std::vector<float> _anchors; // Anchors to be used for encoding, as the array of floats is in the ltrb format of size 8732x4
    size_t _num_anchors;       // number of bbox anchors
    float _criteria = 0.5; // Threshold IoU for matching bounding boxes with anchors. The value needs to be between 0 and 1.
    float _scale; // Rescales the box and anchor values before the offset is calculated (for example, to return to the absolute values).
    bool _offset; // Returns normalized offsets ((encoded_bboxes*scale - anchors*scale) - mean) / stds in EncodedBBoxes that use std and the mean and scale arguments if offset="True"
    std::vector<float> _means, _stds; //_means:  [x y w h] mean values for normalization _stds: [x y w h] standard deviations for offset normalization.
#if ENABLE_HIP
    BoxEncoderGpu *_box_encoder_gpu = nullptr;
#endif
    std::vector<std::vector<size_t>> _sequence_start_framenum_vec; //!< Stores the starting frame number of the sequences.
    std::vector<std::vector<std::vector<float>>>_sequence_frame_timestamps_vec; //!< Stores the timestamps of the frames in a sequences.
    size_t _sequence_batch_size = 0; //!< Indicates the _user_batch_size when sequence reader outputs are required
    size_t _sequence_batch_ratio; //!< Indicates the _user_to_internal_batch_ratio when sequence reader outputs are required
    bool _is_sequence_reader_output = false; //!< Set to true if Sequence Reader is invoked.
    TimingDBG _rb_block_if_empty_time, _rb_block_if_full_time;
};

template <typename T>
std::shared_ptr<T> MasterGraph::add_node(const std::vector<rocalTensor *> &inputs, const std::vector<rocalTensor *> &outputs)
{
    auto node = std::make_shared<T>(inputs, outputs);
    _nodes.push_back(node);

    for(auto& input: inputs)
    {
        if (_tensor_map.find(input) == _tensor_map.end())
            THROW("Input image is invalid, cannot be found among output of previously created nodes")

        auto parent_node = _tensor_map.find(input)->second;
        // parent_node->add_next(node);
        // node->add_previous(parent_node);
    }

    for(auto& output: outputs)
        _tensor_map.insert(make_pair(output, node));

    return node;
}

template <typename T, typename M>
std::shared_ptr<T> MasterGraph::meta_add_node(std::shared_ptr<M> node)
{
    auto meta_node = std::make_shared<T>();
    _meta_data_graph->_meta_nodes.push_back(meta_node);
    meta_node->_node = node;
    meta_node->_batch_size = _user_batch_size;
    // meta_node->_batch_size = _is_sequence_reader_output ? _sequence_batch_size : _user_batch_size;
    return meta_node;
}

/*
 * Explicit specialization for ImageLoaderNode
 */
template<> inline std::shared_ptr<ImageLoaderNode> MasterGraph::add_node(const std::vector<rocalTensor*>& inputs, const std::vector<rocalTensor*>& outputs)
{
    if(_loader_module)
        THROW("A loader already exists, cannot have more than one loader")
    auto node = std::make_shared<ImageLoaderNode>(outputs[0], _device.resources());
    _loader_module = node->get_loader_module();
    _loader_module->set_prefetch_queue_depth(_prefetch_queue_depth);
    _root_nodes.push_back(node);
    for(auto& output: outputs)
        _tensor_map.insert(make_pair(output, node));

    return node;
}

template<> inline std::shared_ptr<ImageLoaderSingleShardNode> MasterGraph::add_node(const std::vector<rocalTensor*>& inputs, const std::vector<rocalTensor*>& outputs)
{
    if(_loader_module)
        THROW("A loader already exists, cannot have more than one loader")
    auto node = std::make_shared<ImageLoaderSingleShardNode>(outputs[0], _device.resources());
    _loader_module = node->get_loader_module();
    _loader_module->set_prefetch_queue_depth(_prefetch_queue_depth);
    _root_nodes.push_back(node);
    std::cerr<<"Head Node added!";
    for(auto& output: outputs)
        _tensor_map.insert(make_pair(output, node));

    return node;
}

template<> inline std::shared_ptr<FusedJpegCropNode> MasterGraph::add_node(const std::vector<rocalTensor*>& inputs, const std::vector<rocalTensor*>& outputs)
{
    if(_loader_module)
        THROW("A loader already exists, cannot have more than one loader")
    auto node = std::make_shared<FusedJpegCropNode>(outputs[0], _device.resources());
    _loader_module = node->get_loader_module();
    _loader_module->set_prefetch_queue_depth(_prefetch_queue_depth);
    _loader_module->set_random_bbox_data_reader(_randombboxcrop_meta_data_reader);
    _root_nodes.push_back(node);
    for(auto& output: outputs)
        _tensor_map.insert(make_pair(output, node));

    return node;
}
template<> inline std::shared_ptr<FusedJpegCropSingleShardNode> MasterGraph::add_node(const std::vector<rocalTensor*>& inputs, const std::vector<rocalTensor*>& outputs)
{
    if(_loader_module)
        THROW("A loader already exists, cannot have more than one loader")
    auto node = std::make_shared<FusedJpegCropSingleShardNode>(outputs[0], _device.resources());
    _loader_module = node->get_loader_module();
    _loader_module->set_prefetch_queue_depth(_prefetch_queue_depth);
    _loader_module->set_random_bbox_data_reader(_randombboxcrop_meta_data_reader);
    _root_nodes.push_back(node);
    for(auto& output: outputs)
        _tensor_map.insert(make_pair(output, node));

    return node;
}

/*
 * Explicit specialization for Cifar10LoaderNode
 */
template<> inline std::shared_ptr<Cifar10LoaderNode> MasterGraph::add_node(const std::vector<rocalTensor*>& inputs, const std::vector<rocalTensor*>& outputs)
{
    if(_loader_module)
        THROW("A loader already exists, cannot have more than one loader")
    auto node = std::make_shared<Cifar10LoaderNode>(outputs[0], _device.resources());
    _loader_module = node->get_loader_module();
    _loader_module->set_prefetch_queue_depth(_prefetch_queue_depth);
    // _root_nodes.push_back(node);
    for(auto& output: outputs)
        _tensor_map.insert(make_pair(output, node));

    return node;
}

/*
 * Explicit specialization for AudioLoaderNode
 */
template<> inline std::shared_ptr<AudioLoaderNode> MasterGraph::add_node(const std::vector<rocalTensor*>& inputs, const std::vector<rocalTensor*>& outputs)
{
    if(_loader_module)
        THROW("A loader already exists, cannot have more than one loader")
    auto node = std::make_shared<AudioLoaderNode>(outputs[0], _device.resources());
    _loader_module = node->get_loader_module();
    _loader_module->set_prefetch_queue_depth(_prefetch_queue_depth);
    _root_nodes.push_back(node);
    for(auto& output: outputs)
        _tensor_map.insert(make_pair(output, node));

    return node;
}
template<> inline std::shared_ptr<AudioLoaderSingleShardNode> MasterGraph::add_node(const std::vector<rocalTensor*>& inputs, const std::vector<rocalTensor*>& outputs)
{
    if(_loader_module)
        THROW("A loader already exists, cannot have more than one loader")
    auto node = std::make_shared<AudioLoaderSingleShardNode>(outputs[0], _device.resources());
    _loader_module = node->get_loader_module();
    _loader_module->set_prefetch_queue_depth(_prefetch_queue_depth);
    _root_nodes.push_back(node);
    for(auto& output: outputs)
        _tensor_map.insert(make_pair(output, node));

    return node;
}

/*
 * Explicit specialization for VideoLoaderNode
 */
template<> inline std::shared_ptr<VideoLoaderNode> MasterGraph::add_node(const std::vector<rocalTensor*>& inputs, const std::vector<rocalTensor*>& outputs)
{
    if(_loader_module)
        THROW("A loader already exists, cannot have more than one loader")
    auto node = std::make_shared<VideoLoaderNode>(outputs[0], _device.resources());
    _loader_module = node->get_loader_module();
    _loader_module->set_prefetch_queue_depth(_prefetch_queue_depth);
    _root_nodes.push_back(node);
    for(auto& output: outputs)
        _tensor_map.insert(make_pair(output, node));

    return node;
}

template<> inline std::shared_ptr<VideoLoaderSingleShardNode> MasterGraph::add_node(const std::vector<rocalTensor*>& inputs, const std::vector<rocalTensor*>& outputs)
{
    if(_loader_module)
        THROW("A loader already exists, cannot have more than one loader")
    auto node = std::make_shared<VideoLoaderSingleShardNode>(outputs[0], _device.resources());
    _loader_module = node->get_loader_module();
    _loader_module->set_prefetch_queue_depth(_prefetch_queue_depth);
    _root_nodes.push_back(node);
    for(auto& output: outputs)
        _tensor_map.insert(make_pair(output, node));

    return node;
}

