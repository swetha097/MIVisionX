import rali_pybind as b
from amd.rali.pipeline import Pipeline
import amd.rali.types as types

def file(*inputs, file_root, bytes_per_sample_hint=0, file_list='', initial_fill='', lazy_init='', num_shards=1,
                 pad_last_batch=False, prefetch_queue_depth=1, preserve=False, random_shuffle=False, read_ahead=False,
                 seed=-1, shard_id=0, shuffle_after_epoch=False, skip_cached_images=False, stick_to_shard=False, tensor_init_bytes=1048576, device=None):

    Pipeline._current_pipeline._reader = "labelReader"
    #Output
    labels = []
    kwargs_pybind = {"source_path": file_root}
    label_reader_meta_data = b.labelReader(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (label_reader_meta_data, labels)




