import sys
from amd.rali.plugin.pytorch import RALIClassificationIterator
from amd.rali.pipeline import Pipeline
import amd.rali.ops as ops
import amd.rali.types as types
import amd.rali.fn as fn
import os
import random


def main():
    if len(sys.argv) < 4:
        print('Please pass image_folder cpu/gpu batch_size classification/detection')
        exit(0)
    if(sys.argv[2] == "cpu"):
        _rali_cpu = True
    else:
        _rali_cpu = False
    bs = int(sys.argv[3])
    if(sys.argv[4] == "detection"):
        _rali_bbox = True
    else:
        _rali_bbox = False

    nt = 1
    di = 0
    crop_size = 224
    image_path = sys.argv[1]
    rali_device = 'cpu' if _rali_cpu else 'gpu'
    decoder_device = 'cpu' if _rali_cpu else 'mixed'
    random_seed = random.SystemRandom().randint(0, 2**32 - 1)
    num_classes = len(next(os.walk(image_path))[1])
    print("num_classes:: ", num_classes)

    pipe = Pipeline(batch_size=bs, num_threads=nt, device_id=di,
                    seed=random_seed, rali_cpu=_rali_cpu)
    # pipe = HybridTrainPipe(batch_size=bs, num_threads=nt, device_id=di, data_dir=image_path, crop=crop_size, rali_cpu=_rali_cpu, rali_type=_rali_type)

    with pipe:  # TODO: Need to add oneHotLabels, CMN, CoinFlip
        if _rali_bbox:
            jpegs, labels, bboxes = fn.readers.caffe(
                path=image_path, bbox=_rali_bbox, random_shuffle=True)
        else:
            jpegs, labels = fn.readers.caffe(
                path=image_path, bbox=_rali_bbox, random_shuffle=True)
        images = fn.decoders.image(jpegs, output_type=types.RGB)
        images = fn.resize(images, resize_x=crop_size,
                           resize_y=crop_size, device=rali_device)
        pipe.set_outputs(images)
    # pipe.build() #TODO:Change required for the new API
    data_loader = RALIClassificationIterator(pipe , display=True)

    # Training loop
    for epoch in range(1):  # loop over the dataset multiple times
        print("epoch:: ", epoch)
        if not _rali_bbox:
            for i, (image_batch, labels) in enumerate(data_loader, 0):  # Classification
                sys.stdout.write("\r Mini-batch " + str(i))
                print("Images", image_batch)
                print("Labels", labels)
            data_loader.reset()
        else:
            for i, (image_batch, bboxes, labels) in enumerate(data_loader, 0):  # Detection
                sys.stdout.write("\r Mini-batch " + str(i))
                print("Images", image_batch)
                print("Bboxes", bboxes)
                print("Labels", labels)
            data_loader.reset()
    # print('Finished Training')
    # print('Finished !!')


if __name__ == '__main__':
    main()