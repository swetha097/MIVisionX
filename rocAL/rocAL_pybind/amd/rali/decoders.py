from amd.rali.global_cfg import Node, add_node
import amd.rali.types as types
import rali_pybind as b



def image(*inputs, user_feature_key_map = None, affine=True, bytes_per_sample_hint=0, cache_batch_copy= True, cache_debug = False, cache_size = 0, cache_threshold = 0,
                 cache_type='', device_memory_padding=16777216, host_memory_padding=8388608, hybrid_huffman_threshold= 1000000, output_type = types.RGB,
                 preserve=False, seed=-1, split_stages=False, use_chunk_allocator= False, use_fast_idct = False, device = None):
    print(f'\n inputs:{inputs} ')

    current_node = Node()
    current_node.node_name = "ImageDecoder"
    current_node.submodule_name ="decoders"
    reader = inputs[0].node_name
    current_node.has_input_image = False
    current_node.has_output_image = True
    current_node.augmentation_node = True
    current_node.kwargs ={"user_feature_key_map":user_feature_key_map} #TODO:Complete this later
    if( reader == 'COCOReader'):
        current_node.rali_c_func_call=b.COCO_ImageDecoderShard
        current_node.kwargs_pybind = {
            "source_path": inputs[0].kwargs['file_root'],
            "json_path": inputs[0].kwargs['annotations_file'],
            "color_format": output_type,
            "shard_id": inputs[0].kwargs['shard_id'],
            "num_shards": inputs[0].kwargs['num_shards'],
            'is_output': current_node.is_output,
            "shuffle": inputs[0].kwargs['random_shuffle'],
            "loop": False,
            "decode_size_policy": types.MAX_SIZE,
            "max_width": 0, #TODO: what happens when we give user given size = multiplier * max_decoded_width
            "max_height":0} #TODO: what happens when we give user given size = multiplier * max_decoded_width
    elif reader == "TFRecordReaderClassification" or reader == "TFRecordReaderDetection":
        current_node.rali_c_func_call = b.TF_ImageDecoder
        current_node.kwargs_pybind = {
            "source_path": inputs[0].kwargs['path'],
            "color_format": output_type,
            "num_shards": inputs[0].kwargs['num_shards'],
            'is_output': current_node.is_output,
            "user_key_for_encoded": user_feature_key_map["image/encoded"],
            "user_key_for_filename": user_feature_key_map["image/filename"],
            "shuffle": inputs[0].kwargs['random_shuffle'],
            "loop": False,
            "decode_size_policy": types.USER_GIVEN_SIZE,
            "max_width": 2000, # TODO: Needs change
            "max_height": 2000} # TODO: Needs change
    elif reader == "Caffe2Reader" or reader == "Caffe2ReaderDetection":
        current_node.rali_c_func_call=b.Caffe2_ImageDecoderShard
        current_node.kwargs_pybind = {
            "source_path": inputs[0].kwargs['path'],
            "color_format": output_type,
            "shard_id": inputs[0].kwargs['shard_id'],
            "num_shards": inputs[0].kwargs['num_shards'],
            'is_output': current_node.is_output,
            "shuffle": inputs[0].kwargs['random_shuffle'],
            "loop": False,
            "decode_size_policy": types.MAX_SIZE,
            "max_width": 0, 
            "max_height":0} 
    elif reader == "CaffeReader" or reader == "CaffeReaderDetection":
        current_node.rali_c_func_call=b.Caffe_ImageDecoderShard
        current_node.kwargs_pybind = {
            "source_path": inputs[0].kwargs['path'],
            "color_format": output_type,
            "shard_id": inputs[0].kwargs['shard_id'],
            "num_shards": inputs[0].kwargs['num_shards'],
            'is_output': current_node.is_output,
            "shuffle": inputs[0].kwargs['random_shuffle'],
            "loop": False,
            "decode_size_policy": types.MAX_SIZE,
            "max_width": 0, 
            "max_height":0} 

    #Connect the Prev Node(inputs[0]) < === > Current Node
    add_node(inputs[0],current_node)
    print(current_node)
    return (current_node)



def image_random_crop(*inputs,user_feature_key_map=None , affine=True, bytes_per_sample_hint=0, device_memory_padding= 16777216, host_memory_padding = 8388608, hybrid_huffman_threshold = 1000000,
                 num_attempts=10, output_type=types.RGB, preserve=False, random_area = None, random_aspect_ratio = None,
                 seed=1, split_stages=False, use_chunk_allocator=False, use_fast_idct= False, device = None):

    #Creating 2 Nodes here (Image Decoder + Random Crop Node)
    #Node 1 Image Decoder
    current_node = Node()
    current_node.node_name = "ImageDecoder"
    current_node.submodule_name ="decoders"
    reader = inputs[0].node_name
    current_node.has_input_image = False
    current_node.has_output_image = True
    current_node.augmentation_node = True
    current_node.kwargs ={"user_feature_key_map":user_feature_key_map} #TODO:Complete this later
    if( reader == 'COCOReader'):
        current_node.rali_c_func_call=b.COCO_ImageDecoderShard
        current_node.kwargs_pybind = {
            "source_path": inputs[0].kwargs['file_root'],
            "json_path": inputs[0].kwargs['annotations_file'],
            "color_format": output_type,
            "shard_id": inputs[0].kwargs['shard_id'],
            "num_shards": inputs[0].kwargs['num_shards'],
            'is_output': current_node.is_output,
            "shuffle": inputs[0].kwargs['random_shuffle'],
            "loop": False,
            "decode_size_policy": types.MAX_SIZE,
            "max_width": 0, #TODO: what happens when we give user given size = multiplier * max_decoded_width
            "max_height":0} #TODO: what happens when we give user given size = multiplier * max_decoded_width
    elif reader == "TFRecordReaderClassification" or reader == "TFRecordReaderDetection":
        current_node.rali_c_func_call = b.TF_ImageDecoder
        current_node.kwargs_pybind = {
            "source_path": inputs[0].kwargs['path'],
            "color_format": output_type,
            "num_shards": inputs[0].kwargs['num_shards'],
            'is_output': current_node.is_output,
            "user_key_for_encoded": user_feature_key_map["image/encoded"],
            "user_key_for_filename": user_feature_key_map["image/filename"],
            "shuffle": inputs[0].kwargs['random_shuffle'],
            "loop": False,
            "decode_size_policy": types.MAX_SIZE,
            "max_width": 0,
            "max_height": 0}
    elif reader == "Caffe2Reader" or reader == "Caffe2ReaderDetection":
        current_node.rali_c_func_call=b.Caffe2_ImageDecoderShard
        current_node.kwargs_pybind = {
            "source_path": inputs[0].kwargs['path'],
            "color_format": output_type,
            "shard_id": inputs[0].kwargs['shard_id'],
            "num_shards": inputs[0].kwargs['num_shards'],
            'is_output': current_node.is_output,
            "shuffle": inputs[0].kwargs['random_shuffle'],
            "loop": False,
            "decode_size_policy": types.MAX_SIZE,
            "max_width": 0, 
            "max_height":0} 
    elif reader == "CaffeReader" or reader == "CaffeReaderDetection":
        current_node.rali_c_func_call=b.Caffe_ImageDecoderShard
        current_node.kwargs_pybind = {
            "source_path": inputs[0].kwargs['path'],
            "color_format": output_type,
            "shard_id": inputs[0].kwargs['shard_id'],
            "num_shards": inputs[0].kwargs['num_shards'],
            'is_output': current_node.is_output,
            "shuffle": inputs[0].kwargs['random_shuffle'],
            "loop": False,
            "decode_size_policy": types.MAX_SIZE,
            "max_width": 0, 
            "max_height":0} 

    #Connect the Prev Node(inputs[0]) < === > Current Node
    add_node(inputs[0],current_node)

    #Node 2 Random Crop
    next_node = Node()
    next_node.node_name = "RandomCrop"
    next_node.submodule_name ="decoders"
    reader = inputs[0].node_name
    next_node.has_input_image = True
    next_node.has_output_image = True
    next_node.augmentation_node = True
    next_node.kwargs ={"user_feature_key_map":user_feature_key_map} #TODO:Complete this later
    next_node.rali_c_func_call=b.Crop
    next_node.kwargs_pybind = {
        "input_image0": current_node.output_image,
        'is_output': next_node.is_output,
        "crop_width": None,
        "crop_height": None,
        "crop_depth": None,
        "crop_pox_x": None,
        "crop_pos_y": None,
        "crop_pox_z": None
    }


    #Connect the Prev Node(current node) < === > next node
    add_node(current_node,next_node)

    return (next_node)


def image_slice(*inputs,affine = True, axes = None, axis_names = "WH",bytes_per_sample_hint = 0, device_memory_padding = 16777216,
                device_memory_padding_jpeg2k = 0, host_memory_padding = 8388608,
                host_memory_padding_jpeg2k = 0, hybrid_huffman_threshold = 1000000,
                 memory_stats = False, normalized_anchor = True, normalized_shape = True, output_type = types.RGB,
                preserve = False, seed = -1, split_stages = False, use_chunk_allocator = False, use_fast_idct = False,device = None):

    #Node 1 Image Decoder
    current_node = Node()
    current_node.node_name = "ImageDecoderSlice"
    current_node.submodule_name ="decoders"
    reader = inputs[0].node_name
    current_node.has_input_image = False
    current_node.has_output_image = True
    current_node.augmentation_node = True
    current_node.kwargs ={"affine":affine} #TODO:Complete this later
    #Reader -> Randon BBox Crop -> ImageDecoderSlice
    if( reader == 'COCOReader'):
        current_node.rali_c_func_call=b.COCO_ImageDecoderSliceShard
        current_node.kwargs_pybind = {
            "source_path": inputs[0].kwargs['file_root'],
            "json_path": inputs[0].kwargs['annotations_file'],
            "color_format": output_type,
            "shard_id": inputs[0].kwargs['shard_id'],
            "shard_count": inputs[0].kwargs['num_shards'],
            'is_output': current_node.is_output,
            "shuffle": inputs[0].kwargs['random_shuffle'],
            "loop": False,
            "decode_size_policy": types.MAX_SIZE,
            "max_width": 0, #TODO: what happens when we give user given size = multiplier * max_decoded_width
            "max_height":0,
            "area_factor": None,
            "aspect_ratio": None,
            "x_drift_factor": None,
            "y_drift_factor": None} #TODO: what happens when we give user given size = multiplier * max_decoded_width


    #Connect the Prev Node(inputs[0]) < === > Current Node
    for i in range(len(inputs)):
        if(isinstance(inputs[i], Node)):
            add_node(inputs[i],current_node)

    return (current_node)
