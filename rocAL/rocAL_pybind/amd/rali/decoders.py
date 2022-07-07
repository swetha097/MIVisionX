import amd.rali.types as types
import rali_pybind as b
from amd.rali.pipeline import Pipeline

def image(*inputs, user_feature_key_map = None, path='', file_root ='', annotations_file= '', shard_id = 0, num_shards = 1, random_shuffle = False, affine=True, bytes_per_sample_hint=0, cache_batch_copy= True, cache_debug = False, cache_size = 0, cache_threshold = 0,
                 cache_type='', device_memory_padding=16777216, host_memory_padding=8388608, hybrid_huffman_threshold= 1000000, output_type = types.RGB,
                 preserve=False, seed=-1, split_stages=False, use_chunk_allocator= False, use_fast_idct = False, device = None):
    reader = Pipeline._current_pipeline._reader

    kwargs_pybind = {
            "source_path": file_root,
            "color_format": output_type,
            "shard_id": shard_id,
            "num_shards": num_shards,
            'is_output': False,
            "shuffle": random_shuffle,
            "loop": False,
            "decode_size_policy": types.USER_GIVEN_SIZE,
            "max_width": 2000,
            "max_height":2000}
    decoded_image = b.ImageDecoderShard(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))

    return (decoded_image)




