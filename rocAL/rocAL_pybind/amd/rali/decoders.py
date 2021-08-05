from amd.rali.global_cfg import Node, add_node
import amd.rali.types as types
import rali_pybind as b



def image(*inputs, **kwargs):
    print(f'\n inputs:{inputs} \n kwargs: {kwargs}')
    current_node = Node()
    current_node.node_name = "ImageDecoder"
    current_node.submodule_name ="decoders"
    reader = inputs[0].node_name
    current_node.has_input_image = False
    current_node.has_output_image = True
    current_node.augmentation_node = True
    if( reader == 'COCOReader'):
        current_node.rali_c_func_call=b.COCO_ImageDecoderShard
        current_node.kwargs_pybind = {
            "source_path": inputs[0].kwargs['file_root'],
            "json_path": inputs[0].kwargs['annotations_file'],
            "color_format": kwargs["output_type"],
            "shard_id": inputs[0].kwargs['shard_id'],
            "num_shards": inputs[0].kwargs['num_shards'],
            'is_output': current_node.is_output,
            "shuffle": inputs[0].kwargs['random_shuffle'],
            "loop": False,
            "decode_size_policy": types.MAX_SIZE,
            "max_width": 0, #Ask Rajy about this when we give user given size = multiplier * max_decoded_width
            "max_height":0} #Ask Rajy about this when we give user given size = multiplier * max_decoded_width
        # current_node.kwargs_pybind["source_path"]= inputs[0].kwargs["file_root"]
        # current_node.kwargs_pybind["json_path"]=inputs[0].kwargs['annotations_file']
        # current_node.kwargs_pybind["color_format"]=kwargs["output_type"]
        # current_node.kwargs_pybind["shard_id"]=inputs[0].kwargs['shard_id']
        # current_node.kwargs_pybind["num_shards"]=inputs[0].kwargs['num_shards']
        # current_node.kwargs_pybind['is_output'] = current_node.is_output
        # current_node.kwargs_pybind["shuffle"]=inputs[0].kwargs['random_shuffle']

        
        # current_node.kwargs_pybind["decode_size_policy"]=types.MAX_SIZE # How to handle this ? (should i store and send the multiplier value ?)
        # current_node.kwargs_pybind["max_width"] = 4 * 300 # Take a look at this
        # current_node.kwargs_pybind["max_height"] = 4 * 300 # Take a look at this

    #Connect the Prev Node(inputs[0]) < === > Current Node
    add_node(inputs[0],current_node)
    print(current_node)
    return (current_node)
