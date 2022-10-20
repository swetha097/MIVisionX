import rocal_pybind as b
from amd.rocal.pipeline import Pipeline
import amd.rocal.types as types

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
    kwargs_pybind = {"source_path": annotations_file, "is_output":True, "mask":False, "is_box_encoder":is_box_encoder }
    b.setSeed(seed)
    meta_data = b.COCOReader(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (meta_data, labels, bboxes)

def video(*inputs,sequence_length, additional_decode_surfaces=2, bytes_per_sample_hint=0, channels=3, dont_use_mmap=False, dtype=types.FLOAT, enable_frame_num=False,  enable_timestamps=False, file_list="", file_list_frame_num=False, file_list_include_preceding_frame=False, file_root="", filenames=[], image_type=types.RGB,
                 initial_fill=1024, labels="", lazy_init=False, normalized=False,
                 num_shards=1, pad_last_batch=False, pad_sequences=False, prefetch_queue_depth=1, preserve=False,
                 random_shuffle=False, read_ahead=False, seed=-1, shard_id=0, skip_cached_images=False, skip_vfr_check=False,step=1,stick_to_shard=False, stride=1, tensor_init_bytes = 1048576, decoder_mode = types.SOFTWARE_DECODE, device=None, name=None):

    Pipeline._current_pipeline._reader = "VideoDecoder"
    #Output
    videos = []
    kwargs_pybind_reader = {"source_path": file_root,"sequence_length":sequence_length,"frame_step":step,"frame_stride":stride,"file_list_frame_num":file_list_frame_num} #VideoMetaDataReader
    b.VideoMetaDataReader(Pipeline._current_pipeline._handle ,*(kwargs_pybind_reader.values()))
    kwargs_pybind_decoder = {"source_path": file_root,"color_format":image_type,"decoder_mode":decoder_mode,"shard_count":num_shards,"sequence_length":sequence_length,"shuffle":random_shuffle ,"is_output":True,"loop":False, "frame_step":step,"frame_stride":stride, "file_list_frame_num":file_list_frame_num } #VideoDecoder

    videos = b.VideoDecoder(Pipeline._current_pipeline._handle ,*(kwargs_pybind_decoder.values()))
    return (videos)

'''
def video_resize(*inputs,sequence_length, resize_width, resize_height, additional_decode_surfaces=2, bytes_per_sample_hint=0, channels=3, dont_use_mmap=False, dtype=types.FLOAT, enable_frame_num=False,  enable_timestamps=False, file_list="", file_list_frame_num=False, file_list_include_preceding_frame=False, file_root="", filenames=[], image_type=types.RGB,
                 initial_fill=1024, labels="", lazy_init=False, normalized=False,
                 num_shards=1, pad_last_batch=False, pad_sequences=False, prefetch_queue_depth=1, preserve=False,
                 random_shuffle=False, read_ahead=False, seed=-1, shard_id=0, skip_cached_images=False, skip_vfr_check=False,step=3,stick_to_shard=False, stride=3, tensor_init_bytes = 1048576, decoder_mode = types.SOFTWARE_DECODE, device=None, name=None):

    Pipeline._current_pipeline._reader = "VideoDecoderResize"
    #Output
    videos = []
    kwargs_pybind_reader = {"source_path": file_root,"sequence_length":sequence_length,"frame_step":step,"frame_stride":stride,"file_list_frame_num":file_list_frame_num} #VideoMetaDataReader
    meta_data = b.VideoMetaDataReader(Pipeline._current_pipeline._handle ,*(kwargs_pybind_reader.values()))
    kwargs_pybind_decoder = {"source_path": file_root,"color_format":image_type,"decoder_mode":decoder_mode,"shard_count":num_shards,"sequence_length":sequence_length,"resize_width":resize_width, "resize_height":resize_height,"shuffle":random_shuffle ,"is_output":False,"loop":False, "frame_step":step,"frame_stride":stride, "file_list_frame_num":file_list_frame_num } #VideoDecoder

    videos = b.VideoDecoderResize(Pipeline._current_pipeline._handle ,*(kwargs_pybind_decoder.values()))
    return (videos, meta_data)
'''
def sequence_reader(*inputs, file_root, sequence_length, bytes_per_sample_hint=0, dont_use_mmap=False, image_type=types.RGB, initial_fill='', lazy_init='', num_shards=1,
                 pad_last_batch=False, prefetch_queue_depth=1, preserve=False, random_shuffle=False, read_ahead=False,
                 seed=-1, shard_id=0, skip_cached_images=False, step = 3, stick_to_shard=False, stride=1, tensor_init_bytes=1048576, device=None):

    Pipeline._current_pipeline._reader = "SequenceReader"
    #Output
    kwargs_pybind = {"source_path": file_root,"color_format":image_type, "shard_count":num_shards, "sequence_length":sequence_length, "is_output":False, "shuffle":random_shuffle, "loop":False, "frame_step":step,"frame_stride":stride}
    frames = b.SequenceReader(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (frames)



