# Copyright (c) 2018 - 2023 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import sys
from amd.rocal.plugin.pytorch import ROCALClassificationIterator
from amd.rocal.pipeline import Pipeline
import amd.rocal.types as types
import amd.rocal.fn as fn
import os
import cv2
from parse_config import parse_args


def draw_patches(img, idx, bboxes=None):
    # image is expected as a tensor, bboxes as tensors
    args = parse_args()
    if args.rocal_gpu:
        image = img.cpu().numpy()
    else:
        image = img.detach().numpy()
    if not args.NHWC:
        image = image.transpose([1, 2, 0])
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if args.classification:
        cv2.imwrite("OUTPUT_IMAGES_PYTHON/CAFFE_READER/CLASSIFICATION/" +
                    str(idx)+"_"+"train"+".png", image)
    else:
        if bboxes is not None:
            bboxes = bboxes.detach().numpy()
            for (l, t, r, b) in bboxes:
                loc_ = [l, t, r, b]
                color = (255, 0, 0)
                thickness = 2
                image = cv2.rectangle(image, (int(loc_[0]), int(loc_[1])), (int(
                    (loc_[2])), int((loc_[3]))), color, thickness)
        cv2.imwrite("OUTPUT_IMAGES_PYTHON/CAFFE_READER/DETECTION/" +
                    str(idx)+"_"+"train"+".png", image)


def main():
    args = parse_args()
    # Args
    image_path = args.image_dataset_path
    rocal_cpu = False if args.rocal_gpu else True
    batch_size = args.batch_size
    rocal_bbox = False if args.classification else True
    num_threads = args.num_threads
    local_rank = args.local_rank
    world_size = args.world_size
    random_seed = args.seed
    device = "gpu" if args.rocal_gpu else "cpu"
    tensor_layout = types.NHWC if args.NHWC else types.NCHW
    num_classes = len(next(os.walk(image_path))[1])
    try:
        if args.classification:
            path = "OUTPUT_IMAGES_PYTHON/CAFFE_READER/CLASSIFICATION/"
        else:
            path = "OUTPUT_IMAGES_PYTHON/CAFFE_READER/DETECTION/"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
    except OSError as error:
        print(error)
    print("num_classes:: ", num_classes)
    # Create Pipeline instance
    pipe = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=args.local_rank, seed=random_seed, rocal_cpu=rocal_cpu)
    # Use pipeline instance to make calls to reader, decoder & augmentation's
    with pipe:
        if rocal_bbox:
            jpegs, labels, bboxes = fn.readers.caffe(path=image_path, bbox=rocal_bbox)
            images = fn.decoders.image(jpegs, path=image_path, shard_id=local_rank, random_shuffle=True)

        else:
            jpegs, labels = fn.readers.caffe(path=image_path, bbox=rocal_bbox)
            images = fn.decoders.image(jpegs, path=image_path, output_type=types.RGB, shard_id=local_rank, num_shards=world_size, random_shuffle=True)

        images = fn.resize(images, resize_width=224, resize_height=224, output_layout=tensor_layout)
        pipe.set_outputs(images)
    # Build the pipeline
    pipe.build()
    # Dataloader
    data_loader = ROCALClassificationIterator(pipe, display=0, device=device, device_id=args.local_rank)
    # Training loop
    cnt = 0
    # Enumerate over the Dataloader
    for epoch in range(args.num_epochs):  # loop over the dataset multiple times
        print("epoch:: ", epoch)
        if not rocal_bbox:
            for i, ([image_batch], labels) in enumerate(data_loader, 0):  # Classification
                if args.print_tensor:
                    sys.stdout.write("\r Mini-batch " + str(i))
                    print("Images", image_batch)
                    print("Labels", labels)
                for element in list(range(batch_size)):
                    cnt = cnt + 1
                    draw_patches(image_batch[element], cnt)
            data_loader.reset()
        else:
            for i, ([image_batch], bboxes, labels) in enumerate(data_loader, 0):  # Detection
                if i == 0:
                    if args.print_tensor:
                        sys.stdout.write("\r Mini-batch " + str(i))
                        print("Images", image_batch)
                        print("Bboxes", bboxes)
                        print("Labels", labels)
                for element in list(range(batch_size)):
                    cnt = cnt + 1
                    draw_patches(image_batch[element], cnt, bboxes[element])
            data_loader.reset()
    print("##############################  CAFFE READER (CLASSIFCATION/ DETECTION)  SUCCESS  ############################")

if __name__ == "__main__":
    main()
