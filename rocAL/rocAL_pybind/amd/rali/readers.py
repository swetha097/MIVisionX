from amd.rali.global_cfg import Node, add_node
import rali_pybind as b


def coco(*inputs,file_root, annotations_file='', bytes_per_sample_hint=0, dump_meta_files=False, dump_meta_files_path='', file_list='', initial_fill=1024,  lazy_init=False, ltrb=False, masks=False, meta_files_path='', num_shards=1, pad_last_batch=False, prefetch_queue_depth=1,
                 preserve=False, random_shuffle=False, ratio=False, read_ahead=False,
                 save_img_ids=False, seed=-1, shard_id=0, shuffle_after_epoch=False, size_threshold=0.1,
                 skip_cached_images=False, skip_empty=False, stick_to_shard=False, tensor_init_bytes=1048576):
    
    print(f'\n args:{inputs}')
    #Output
    labels = []
    bboxes = []
    kwargs_pybind = {"source_path": annotations_file, "is_output":False}
    #Node Object
    current_node=Node()
    current_node.prev.append("NULL")
    current_node.node_name = "COCOReader"
    current_node.submodule_name ="readers"
    current_node.rali_c_func_call = b.COCOReader
    current_node.kwargs_pybind = kwargs_pybind
    current_node.kwargs = {"file_root": file_root, "annotations_file": annotations_file, "bytes_per_sample_hint": bytes_per_sample_hint,
                           "dump_meta_files": dump_meta_files, "dump_meta_files_path": dump_meta_files_path, "file_list": file_list, "initial_fill": initial_fill,  "lazy_init": lazy_init, "ltrb": ltrb, "masks": masks, "meta_files_path": meta_files_path, "num_shards": num_shards, "pad_last_batch": pad_last_batch, "prefetch_queue_depth": prefetch_queue_depth,
                           "preserve": preserve, "random_shuffle": random_shuffle, "ratio": ratio, "read_ahead": read_ahead,
                           "save_img_ids": save_img_ids, "seed": seed, "shard_id": shard_id, "shuffle_after_epoch": shuffle_after_epoch, "size_threshold": size_threshold,
                           "skip_cached_images": skip_cached_images, "skip_empty": skip_empty, "stick_to_shard": stick_to_shard, "tensor_init_bytes": tensor_init_bytes}
    current_node.has_input_image = False
    current_node.has_output_image = False
    return (current_node, labels, bboxes)


