from amd.rali.global_cfg import Node, add_node
import rali_pybind as b
from amd.rali.pipeline import Pipeline
import amd.rali.types as types


def coco_trail(*inputs,file_root, annotations_file='', bytes_per_sample_hint=0, dump_meta_files=False, dump_meta_files_path='', file_list='', initial_fill=1024,  lazy_init=False, ltrb=False, masks=False, meta_files_path='', num_shards=1, pad_last_batch=False, prefetch_queue_depth=1,
                 preserve=False, random_shuffle=False, ratio=False, read_ahead=False,
                 save_img_ids=False, seed=-1, shard_id=0, shuffle_after_epoch=False, size_threshold=0.1,
                 skip_cached_images=False, skip_empty=False, stick_to_shard=False, tensor_init_bytes=1048576):

    #Output
    labels = []
    bboxes = []
    kwargs_pybind = {"source_path": annotations_file, "is_output":True}
    print("\n Pipeline handle in reader:", Pipeline._current_pipeline._handle)
    meta_data = b.COCOReader(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    Pipeline._current_pipeline._reader = "COCOReader"

    return (meta_data, labels, bboxes)

def coco(*inputs,file_root, annotations_file='', bytes_per_sample_hint=0, dump_meta_files=False, dump_meta_files_path='', file_list='', initial_fill=1024,  lazy_init=False, ltrb=False, masks=False, meta_files_path='', num_shards=1, pad_last_batch=False, prefetch_queue_depth=1,
                 preserve=False, random_shuffle=False, ratio=False, read_ahead=False,
                 save_img_ids=False, seed=-1, shard_id=0, shuffle_after_epoch=False, size_threshold=0.1,
                 skip_cached_images=False, skip_empty=False, stick_to_shard=False, tensor_init_bytes=1048576):

    Pipeline._current_pipeline._reader = "COCOReader"
    #Output
    labels = []
    bboxes = []
    kwargs_pybind = {"source_path": annotations_file, "is_output":True}
    b.setSeed(seed)
    meta_data = b.COCOReader(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (meta_data, labels, bboxes)


def file(*inputs, file_root, bytes_per_sample_hint=0, file_list='', initial_fill='', lazy_init='', num_shards=1,
                 pad_last_batch=False, prefetch_queue_depth=1, preserve=False, random_shuffle=False, read_ahead=False,
                 seed=-1, shard_id=0, shuffle_after_epoch=False, skip_cached_images=False, stick_to_shard=False, tensor_init_bytes=1048576, device=None):

    Pipeline._current_pipeline._reader = "labelReader"
    #Output
    labels = []
    kwargs_pybind = {"source_path": file_root}
    label_reader_meta_data = b.labelReader(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (label_reader_meta_data, labels)



def tfrecord(*inputs, path, user_feature_key_map, features, index_path="", reader_type=0, bytes_per_sample_hint=0, initial_fill=1024, lazy_init=False,
 num_shards=1, pad_last_batch=False, prefetch_queue_depth=1, preserve=False, random_shuffle=False, read_ahead=False, seed=-1, shard_id=0, skip_cached_images=False, stick_to_shard=False, tensor_init_bytes=1048576,  device=None):



    labels=[]
    if reader_type == 1:
        Pipeline._current_pipeline._reader = "TFRecordReaderDetection"
        for key in features.keys():
            if key not in user_feature_key_map.keys():
                    print(
                        "For Object Detection, RALI TFRecordReader needs all the following keys in the featureKeyMap:")
                    print("image/encoded\nimage/class/label\nimage/class/text\nimage/object/bbox/xmin\nimage/object/bbox/ymin\nimage/object/bbox/xmax\nimage/object/bbox/ymax\n")
                    exit()
        tf_meta_data = b.TFReaderDetection(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
        kwargs_pybind = {"path": path, "is_output": True, "user_key_for_label": user_feature_key_map["image/class/label"], "user_key_for_text": user_feature_key_map["image/class/text"], "user_key_for_xmin": user_feature_key_map["image/object/bbox/xmin"],
                         "user_key_for_ymin": user_feature_key_map["image/object/bbox/ymin"], "user_key_for_xmax": user_feature_key_map["image/object/bbox/xmax"], "user_key_for_ymax": user_feature_key_map["image/object/bbox/ymax"], "user_key_for_filename": user_feature_key_map["image/filename"]}
    else:
        Pipeline._current_pipeline._reader = "TFRecordReaderClassification"
        for key in features.keys():
                if key not in user_feature_key_map.keys():
                    print(
                        "For Image Classification, RALI TFRecordReader needs all the following keys in the featureKeyMap:")
                    print("image/encoded\nimage/class/label\n")
                exit()
        tf_meta_data = b.TFReader(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
        kwargs_pybind = {"path": path, "is_output": True, "user_key_for_label": user_feature_key_map[
            "image/class/label"], "user_key_for_filename": user_feature_key_map["image/filename"]}

    features["image/encoded"] = tf_meta_data
    features["image/class/label"] = labels
    return features

def caffe(*inputs, path, bbox=False, bytes_per_sample_hint=0, image_available=True, initial_fill=1024, label_available=True,
                 lazy_init=False,  num_shards=1,
                 pad_last_batch=False, prefetch_queue_depth=1, preserve=False, random_shuffle=False, read_ahead=False,
                 seed=-1, shard_id=0, skip_cached_images=False, stick_to_shard=False, tensor_init_bytes=1048576, device=None):

    #Output
    bboxes = []
    labels = []
    kwargs_pybind = {"source_path": path}
    #Node Object
    if (bbox == True):
        Pipeline._current_pipeline._reader = "CaffeReaderDetection"
        caffe_reader_meta_data = b.CaffeReaderDetection(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    else:
        Pipeline._current_pipeline._reader = "CaffeReader"
        caffe_reader_meta_data = b.CaffeReader(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))

    if (bbox == True):
        return (caffe_reader_meta_data,bboxes, labels)
    else:
        return (caffe_reader_meta_data, labels)


def caffe2(*inputs, path, bbox=False, additional_inputs=0, bytes_per_sample_hint=0, image_available=True, initial_fill=1024, label_type=0,
                 lazy_init=False, num_labels=1,  num_shards=1,
                 pad_last_batch=False, prefetch_queue_depth=1, preserve=False, random_shuffle=False, read_ahead=False,
                 seed=-1, shard_id=0, skip_cached_images=False, stick_to_shard=False, tensor_init_bytes=1048576, device=None):

    #Output
    bboxes = []
    labels = []
    kwargs_pybind = {"source_path": path, "is_output":True}
    if (bbox == True):
        Pipeline._current_pipeline._reader = "Caffe2ReaderDetection"
        caffe2_meta_data = b.Caffe2ReaderDetection(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    else:
        Pipeline._current_pipeline._reader = "Caffe2Reader"
        caffe2_meta_data = b.Caffe2Reader(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    if (bbox == True):
        return (caffe2_meta_data,bboxes, labels)
    else:
        return (caffe2_meta_data, labels)

def video(*inputs,sequence_length, additional_decode_surfaces=2, bytes_per_sample_hint=0, channels=3, dont_use_mmap=False, dtype=types.FLOAT, enable_frame_num=False,  enable_timestamps=False, file_list="", file_list_frame_num=False, file_list_include_preceding_frame=False, file_root="", filenames=[], image_type=types.RGB,
                 initial_fill=1024, labels="", lazy_init=False, normalized=False,
                 num_shards=1, pad_last_batch=False, pad_sequences=False, prefetch_queue_depth=1, preserve=False,
                 random_shuffle=False, read_ahead=False, seed=-1, shard_id=0, skip_cached_images=False, skip_vfr_check=False,step=3,stick_to_shard=False, stride=1, tensor_init_bytes = 1048576, decoder_mode = types.SOFTWARE_DECODE, device=None, name=None):

    Pipeline._current_pipeline._reader = "VideoDecoder"
    #Output
    videos = []
    kwargs_pybind_reader = {"source_path": file_root,"sequence_length":sequence_length,"frame_step":step,"frame_stride":stride,"file_list_frame_num":file_list_frame_num} #VideoMetaDataReader
    meta_data = b.VideoMetaDataReader(Pipeline._current_pipeline._handle ,*(kwargs_pybind_reader.values()))
    kwargs_pybind_decoder = {"source_path": file_root,"color_format":image_type,"decoder_mode":decoder_mode,"shard_count":num_shards,"sequence_length":sequence_length,"shuffle":random_shuffle ,"is_output":False,"loop":False, "frame_step":step,"frame_stride":stride, "file_list_frame_num":file_list_frame_num } #VideoDecoder

    videos = b.VideoDecoder(Pipeline._current_pipeline._handle ,*(kwargs_pybind_decoder.values()))
    return (videos)
