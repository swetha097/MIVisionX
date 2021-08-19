from amd.rali.global_cfg import Node, add_node
import rali_pybind as b


def coco(*inputs,file_root, annotations_file='', bytes_per_sample_hint=0, dump_meta_files=False, dump_meta_files_path='', file_list='', initial_fill=1024,  lazy_init=False, ltrb=False, masks=False, meta_files_path='', num_shards=1, pad_last_batch=False, prefetch_queue_depth=1,
                 preserve=False, random_shuffle=False, ratio=False, read_ahead=False,
                 save_img_ids=False, seed=-1, shard_id=0, shuffle_after_epoch=False, size_threshold=0.1,
                 skip_cached_images=False, skip_empty=False, stick_to_shard=False, tensor_init_bytes=1048576):
    
    #Output
    labels = []
    bboxes = []
    kwargs_pybind = {"source_path": annotations_file, "is_output":True}
    #Node Object
    current_node=Node()
    current_node.prev.append("NULL")
    current_node.node_name = "COCOReader"
    current_node.submodule_name ="readers"
    current_node.rali_c_func_call = b.COCOReader
    current_node.kwargs_pybind = kwargs_pybind
    current_node.augmentation_node = True
    current_node.kwargs = {"file_root": file_root, "annotations_file": annotations_file, "bytes_per_sample_hint": bytes_per_sample_hint,
                           "dump_meta_files": dump_meta_files, "dump_meta_files_path": dump_meta_files_path, "file_list": file_list, "initial_fill": initial_fill,  "lazy_init": lazy_init, "ltrb": ltrb, "masks": masks, "meta_files_path": meta_files_path, "num_shards": num_shards, "pad_last_batch": pad_last_batch, "prefetch_queue_depth": prefetch_queue_depth,
                           "preserve": preserve, "random_shuffle": random_shuffle, "ratio": ratio, "read_ahead": read_ahead,
                           "save_img_ids": save_img_ids, "seed": seed, "shard_id": shard_id, "shuffle_after_epoch": shuffle_after_epoch, "size_threshold": size_threshold,
                           "skip_cached_images": skip_cached_images, "skip_empty": skip_empty, "stick_to_shard": stick_to_shard, "tensor_init_bytes": tensor_init_bytes}
    current_node.has_input_image = False
    current_node.has_output_image = False
    return (current_node, labels, bboxes)


def file(*inputs, file_root, bytes_per_sample_hint=0, file_list='', initial_fill='', lazy_init='', num_shards=1,
                 pad_last_batch=False, prefetch_queue_depth=1, preserve=False, random_shuffle=False, read_ahead=False,
                 seed=-1, shard_id=0, shuffle_after_epoch=False, skip_cached_images=False, stick_to_shard=False, tensor_init_bytes=1048576, device=None):
       
    #Output
    labels = []
    kwargs_pybind = {"source_path": file_root}
    #Node Object
    current_node=Node()
    current_node.prev.append("NULL")
    current_node.node_name = "FileReader"
    current_node.submodule_name ="readers"
    current_node.rali_c_func_call = b.labelReader
    current_node.kwargs_pybind = kwargs_pybind
    current_node.augmentation_node = True
    current_node.kwargs = {"file_root": file_root, "bytes_per_sample_hint": bytes_per_sample_hint, "file_list": file_list, "initial_fill": initial_fill,  "lazy_init": lazy_init,
                           "num_shards": num_shards, "pad_last_batch": pad_last_batch, "prefetch_queue_depth": prefetch_queue_depth,
                           "preserve": preserve, "random_shuffle": random_shuffle,"read_ahead": read_ahead,
                           "seed": seed, "shard_id": shard_id, "shuffle_after_epoch": shuffle_after_epoch,
                           "skip_cached_images": skip_cached_images, "stick_to_shard": stick_to_shard, "tensor_init_bytes": tensor_init_bytes}
    current_node.has_input_image = False
    current_node.has_output_image = False
    return (current_node, labels)



def tfrecord(*inputs, path, user_feature_key_map, features, index_path="", reader_type=0, bytes_per_sample_hint=0, initial_fill=1024, lazy_init=False,
 num_shards=1, pad_last_batch=False, prefetch_queue_depth=1, preserve=False, random_shuffle=False, read_ahead=False, seed=-1, shard_id=0, skip_cached_images=False, stick_to_shard=False, tensor_init_bytes=1048576,  device=None):
       

    #Node Object
    current_node=Node()
    current_node.prev.append("NULL")
    
    current_node.submodule_name ="readers"
    if reader_type == 1:
        current_node.node_name = "TFRecordReaderDetection"
        current_node.rali_c_func_call = b.TFReaderDetection
        kwargs_pybind = {"path": path, "is_output": True, "user_key_for_label": user_feature_key_map["image/class/label"], "user_key_for_text": user_feature_key_map["image/class/text"], "user_key_for_xmin": user_feature_key_map["image/object/bbox/xmin"],
                         "user_key_for_ymin": user_feature_key_map["image/object/bbox/ymin"], "user_key_for_xmax": user_feature_key_map["image/object/bbox/xmax"], "user_key_for_ymax": user_feature_key_map["image/object/bbox/ymax"], "user_key_for_filename": user_feature_key_map["image/filename"]}
    else:
        current_node.node_name = "TFRecordReaderClassification"
        current_node.rali_c_func_call = b.TFReader
        kwargs_pybind = {"path": path, "is_output": True, "user_key_for_label": user_feature_key_map[
            "image/class/label"], "user_key_for_filename": user_feature_key_map["image/filename"]}

    current_node.kwargs_pybind = kwargs_pybind
    current_node.augmentation_node = True
    current_node.kwargs = {"path": path, "user_feature_key_map": user_feature_key_map, "features": features, "index_path": index_path, "reader_type": reader_type, "bytes_per_sample_hint": bytes_per_sample_hint, "initial_fill": initial_fill,  "lazy_init": lazy_init,
                           "num_shards": num_shards, "pad_last_batch": pad_last_batch, "prefetch_queue_depth": prefetch_queue_depth,
                           "preserve": preserve, "random_shuffle": random_shuffle, "read_ahead": read_ahead,
                           "seed": seed, "shard_id": shard_id, "skip_cached_images": skip_cached_images, "stick_to_shard": stick_to_shard, "tensor_init_bytes": tensor_init_bytes}
    current_node.has_input_image = False
    current_node.has_output_image = False
    return (current_node)

def caffe(*inputs, path, bbox=False, bytes_per_sample_hint=0, image_available=True, initial_fill=1024, label_available=True,
                 lazy_init=False,  num_shards=1,
                 pad_last_batch=False, prefetch_queue_depth=1, preserve=False, random_shuffle=False, read_ahead=False,
                 seed=-1, shard_id=0, skip_cached_images=False, stick_to_shard=False, tensor_init_bytes=1048576, device=None):
       
    #Output
    bboxes = []
    labels = []
    kwargs_pybind = {"source_path": path}
    #Node Object
    current_node=Node()
    current_node.prev.append("NULL")
    if (bbox == True):
        current_node.node_name = "CaffeReaderDetection"
        current_node.rali_c_func_call = b.CaffeReaderDetection
    else:
        current_node.node_name = "CaffeReader"
        current_node.rali_c_func_call = b.CaffeReader

    current_node.submodule_name ="readers"
    current_node.kwargs_pybind = kwargs_pybind
    current_node.augmentation_node = True
    current_node.kwargs = {"path": path,"bbox":bbox, "bytes_per_sample_hint": bytes_per_sample_hint, "image_available": image_available, "initial_fill": initial_fill, "label_available": label_available, "lazy_init": lazy_init,
                           "num_shards": num_shards, "pad_last_batch": pad_last_batch, "prefetch_queue_depth": prefetch_queue_depth,
                           "preserve": preserve, "random_shuffle": random_shuffle,"read_ahead": read_ahead,
                           "seed": seed, "shard_id": shard_id, "skip_cached_images": skip_cached_images, "stick_to_shard": stick_to_shard, "tensor_init_bytes": tensor_init_bytes}
    current_node.has_input_image = False
    current_node.has_output_image = False
    if (bbox == True):
        return (current_node,bboxes, labels)
    else:
        return (current_node, labels)


def caffe2(*inputs, path, bbox=False, additional_inputs=0, bytes_per_sample_hint=0, image_available=True, initial_fill=1024, label_type=0,
                 lazy_init=False, num_labels=1,  num_shards=1,
                 pad_last_batch=False, prefetch_queue_depth=1, preserve=False, random_shuffle=False, read_ahead=False,
                 seed=-1, shard_id=0, skip_cached_images=False, stick_to_shard=False, tensor_init_bytes=1048576, device=None):
       
    #Output
    bboxes = []
    labels = []
    kwargs_pybind = {"source_path": path, "is_output":True}
    #Node Object
    current_node=Node()
    current_node.prev.append("NULL")
    if (bbox == True):
        current_node.node_name = "Caffe2ReaderDetection"
        current_node.rali_c_func_call = b.Caffe2ReaderDetection
    else:
        current_node.node_name = "Caffe2Reader"
        current_node.rali_c_func_call = b.Caffe2Reader

    current_node.submodule_name ="readers"
    current_node.kwargs_pybind = kwargs_pybind
    current_node.augmentation_node = True
    current_node.kwargs = {"path": path,"bbox":bbox, "additional_inputs":additional_inputs, "bytes_per_sample_hint": bytes_per_sample_hint, "image_available": image_available, "initial_fill": initial_fill, "label_type": label_type, "lazy_init": lazy_init, "num_labels":num_labels,
                           "num_shards": num_shards, "pad_last_batch": pad_last_batch, "prefetch_queue_depth": prefetch_queue_depth,
                           "preserve": preserve, "random_shuffle": random_shuffle,"read_ahead": read_ahead,
                           "seed": seed, "shard_id": shard_id, "skip_cached_images": skip_cached_images, "stick_to_shard": stick_to_shard, "tensor_init_bytes": tensor_init_bytes}
    current_node.has_input_image = False
    current_node.has_output_image = False
    if (bbox == True):
        return (current_node,bboxes, labels)
    else:
        return (current_node, labels)

