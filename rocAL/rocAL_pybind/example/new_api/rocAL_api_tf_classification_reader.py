from amd.rocal.plugin.tf import RALIIterator
from amd.rocal.pipeline import Pipeline
import amd.rocal.types as types
import os
import amd.rocal.fn as fn
import tensorflow as tf
import numpy as np
from parse_config import parse_args


def draw_patches(img,idx):
    #image is expected as a tensor, bboxes as numpy
    import cv2
    image = img.transpose([0,1,2])
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("OUTPUT_IMAGES_PYTHON/NEW_API/TF_READER/CLASSIFICATION/" + str(idx)+"_"+"train"+".png", image)

def main():
    args = parse_args()
    # Args
    imagePath = args.image_dataset_path
    raliCPU = False if args.rocal_gpu else True
    batch_size = args.batch_size
    oneHotLabel = 1
    num_threads = args.num_threads
    TFRecordReaderType = 0
    featureKeyMap = {
        'image/encoded':'image/encoded',
        'image/class/label':'image/class/label',
        'image/filename':'image/filename'
    }
    try:
        path= "OUTPUT_IMAGES_PYTHON/NEW_API/TF_READER/CLASSIFICATION/"
        os.makedirs(path)
    except OSError as error:
        print(error)
    # Create Pipeline instance
    pipe = Pipeline(batch_size=batch_size, num_threads=num_threads,device_id=args.local_rank, seed=2, rali_cpu=raliCPU)
    # Use pipeline instance to make calls to reader, decoder & augmentation's
    with pipe:
        inputs = fn.readers.tfrecord(path=imagePath, index_path = "", reader_type=TFRecordReaderType, user_feature_key_map=featureKeyMap,
            features={
                'image/encoded':tf.io.FixedLenFeature((), tf.string, ""),
                'image/class/label':tf.io.FixedLenFeature([1], tf.int64,  -1),
                'image/filename':tf.io.FixedLenFeature((), tf.string, "")
            }
        )
        jpegs = inputs["image/encoded"]
        images = fn.decoders.image(jpegs, user_feature_key_map=featureKeyMap, output_type=types.RGB, path=imagePath)
        resized = fn.resize(images, resize_x=300, resize_y=300)
        if(oneHotLabel == 1):
            labels = inputs["image/class/label"]
            _ = fn.one_hot(labels, num_classes=1000)
        pipe.set_outputs(resized)
    # Build the pipeline
    pipe.build()
    # Dataloader
    imageIterator = RALIIterator(pipe)
    cnt = 0
    # Enumerate over the Dataloader
    for i, (images_array, labels_array) in enumerate(imageIterator, 0):
        images_array = np.transpose(images_array, [0, 2, 3, 1])
        print("\n",i)
        print("lables_array",labels_array)
        for element in list(range(batch_size)):
            cnt = cnt + 1
            draw_patches(images_array[element],cnt)
        print("\n\nPrinted first batch with", (batch_size), "images!")
        break
    imageIterator.reset()



if __name__ == '__main__':
    main()