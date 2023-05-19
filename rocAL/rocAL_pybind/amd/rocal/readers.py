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

def coco(*inputs,file_root='', annotations_file='', bytes_per_sample_hint=0, dump_meta_files=False, dump_meta_files_path='', file_list='', initial_fill=1024,  lazy_init=False, ltrb=False, polygon_mask=False, pixelwise_mask = False, meta_files_path='', num_shards=1, pad_last_batch=False, prefetch_queue_depth=1,
                 preserve=False, random_shuffle=False, ratio=False, read_ahead=False,
                 save_img_ids=False, seed=-1, shard_id=0, shuffle_after_epoch=False, size_threshold=0.1, is_box_encoder=False,
                 skip_cached_images=False, skip_empty=False, stick_to_shard=False, tensor_init_bytes=1048576, is_foreground=False, value=0, is_threshold=True):

    Pipeline._current_pipeline._reader = "COCOReader"
    #Output
    labels = []
    bboxes = []
    pixelwisemask = [] if pixelwise_mask else None
    polygonmask = [] if polygon_mask else None
    print("polyyyyyyyyyy", polygon_mask)
    if pixelwise_mask:
        kwargs_pybind = {"is_foreground": is_foreground, "value":value, "is_threshold":is_threshold}
        b.SetRandomPixelMaskConfig(Pipeline._current_pipeline._handle,*(kwargs_pybind.values()))
    kwargs_pybind = {"source_path": annotations_file, "is_output":True, "polygon_mask":polygon_mask, "is_box_encoder":is_box_encoder, "pixelwise_mask": pixelwise_mask }
    b.setSeed(seed)
    meta_data = b.COCOReader(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    if pixelwise_mask:
        print("pixel")
        return (meta_data, labels, bboxes, pixelwisemask)
    elif polygon_mask:
        print("POLYGON MASK")
        return (meta_data, labels, bboxes, polygonmask)
    else:
        print("ELD}SE")
        return (meta_data, labels, bboxes)




