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

def coco(*inputs,file_root='', annotations_file='', bytes_per_sample_hint=0, dump_meta_files=False, dump_meta_files_path='', file_list='', initial_fill=1024,  lazy_init=False, ltrb=False, masks=False, meta_files_path='', num_shards=1, pad_last_batch=False, prefetch_queue_depth=1,
                 preserve=False, random_shuffle=False, ratio=False, read_ahead=False,
                 save_img_ids=False, seed=-1, shard_id=0, shuffle_after_epoch=False, size_threshold=0.1, is_box_encoder=False,
                 skip_cached_images=False, skip_empty=False, stick_to_shard=False, tensor_init_bytes=1048576):

    Pipeline._current_pipeline._reader = "COCOReader"
    #Output
    labels = []
    bboxes = []
    kwargs_pybind = {"source_path": annotations_file, "is_output":True, "is_box_encoder":is_box_encoder }
    b.setSeed(seed)
    meta_data = b.COCOReader(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (meta_data, labels, bboxes)

def tfrecord(*inputs, path, user_feature_key_map, features, index_path="", reader_type=0, bytes_per_sample_hint=0, initial_fill=1024, lazy_init=False,
 num_shards=1, pad_last_batch=False, prefetch_queue_depth=1, preserve=False, random_shuffle=False, read_ahead=False, seed=1, shard_id=0, skip_cached_images=False, stick_to_shard=False, tensor_init_bytes=1048576,  device=None):
    labels=[]
    b.setSeed(seed)
    seed=-1
    print("seed****************************** ",seed)
    if reader_type == 1:
        Pipeline._current_pipeline._reader = "TFRecordReaderDetection"
        print("user_feature_key_map",user_feature_key_map)
        kwargs_pybind = {"path": path, "is_output": True, "user_key_for_label": user_feature_key_map["image/class/label"], "user_key_for_text": user_feature_key_map["image/class/text"], "user_key_for_xmin": user_feature_key_map["image/object/bbox/xmin"],
                         "user_key_for_ymin": user_feature_key_map["image/object/bbox/ymin"], "user_key_for_xmax": user_feature_key_map["image/object/bbox/xmax"], "user_key_for_ymax": user_feature_key_map["image/object/bbox/ymax"], "user_key_for_filename": user_feature_key_map["image/filename"]}
        for key in features.keys():
            if key not in user_feature_key_map.keys():
                    print(
                        "For Object Detection, ROCAL TFRecordReader needs all the following keys in the featureKeyMap:")
                    print("image/encoded\nimage/class/label\nimage/class/text\nimage/object/bbox/xmin\nimage/object/bbox/ymin\nimage/object/bbox/xmax\nimage/object/bbox/ymax\n")
                    exit()    
        tf_meta_data = b.TFReaderDetection(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))

        print("check 3333")
    else:
        Pipeline._current_pipeline._reader = "TFRecordReaderClassification"
        kwargs_pybind = {"path": path, "is_output": True, "user_key_for_label": user_feature_key_map[
            "image/class/label"], "user_key_for_filename": user_feature_key_map["image/filename"]}
        for key in features.keys():
                if key not in user_feature_key_map.keys():
                    print(
                        "For Image Classification, ROCAL TFRecordReader needs all the following keys in the featureKeyMap:")
                    print("image/encoded\nimage/class/label\n")
                    exit()
        tf_meta_data = b.TFReader(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    features["image/encoded"] = tf_meta_data
    features["image/class/label"] = labels
    return features

